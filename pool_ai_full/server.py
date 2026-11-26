#!/usr/bin/env python3
import base64
from contextlib import asynccontextmanager
import json
import logging
import os
from time import perf_counter
from typing import Annotated, Literal

import anyio
import cv2 as cv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException
import uvicorn

from .analyze import analyze_pool_image
from .image_input import MAX_IMAGE_BYTES, InvalidImage, decode_image

MAX_BODY_BYTES = MAX_IMAGE_BYTES + 64 * 1024
MAX_CONCURRENT_ANALYSES = 2
logger = logging.getLogger("uvicorn.error")


class Shot(BaseModel):
    score: float
    object_idx: int
    pocket: tuple[float, float]
    ghost: tuple[float, float]
    cut_deg: float


class AnalysisResponse(BaseModel):
    advice: str
    outcome: Literal["recommendation", "no_usable_table", "no_cue_ball", "no_clear_pot"]
    rectified: bool
    num_balls: int
    cue_idx: int | None
    best: Shot | None
    overlay_width: int
    overlay_height: int
    overlay_png_base64: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


def error_response(status, code, message):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def body_too_large():
    return error_response(413, "image_too_large", "Multipart request exceeds the upload limit.")


class LimitAnalysisBody:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/analyze":
            await self.app(scope, receive, send)
            return

        length = dict(scope["headers"]).get(b"content-length", b"")
        if length.isdigit() and (len(length) > 16 or int(length) > MAX_BODY_BYTES):
            await body_too_large()(scope, receive, send)
            return

        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > MAX_BODY_BYTES:
                scope.setdefault("state", {})["analysis_body_too_large"] = True
                # Let Starlette close any multipart files created before the limit.
                raise MultiPartException("Multipart request exceeds the upload limit.")
            return message

        await self.app(scope, bounded_receive, send)


@asynccontextmanager
async def lifespan(app):
    app.state.analysis_limiter = anyio.CapacityLimiter(MAX_CONCURRENT_ANALYSES)
    yield


app = FastAPI(title="CueVision Analysis", lifespan=lifespan)
app.add_middleware(LimitAnalysisBody)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    return error_response(422, "validation_error", "Check the image field and query parameters.")


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    if getattr(request.state, "analysis_body_too_large", False):
        return body_too_large()
    if exc.status_code == 400:
        return error_response(400, "invalid_upload", "Could not parse the multipart upload.")
    response = error_response(exc.status_code, "http_error", str(exc.detail))
    response.headers.update(exc.headers or {})
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready", responses={503: {"model": ErrorResponse}})
async def ready():
    if not cv.haveImageWriter(".png"):
        return error_response(503, "not_ready", "The PNG encoder is unavailable.")
    return {"status": "ok"}


def analyze_upload(file, felt, detector):
    started = perf_counter()
    image = decode_image(file.read(MAX_IMAGE_BYTES + 1))
    result = analyze_pool_image(image, felt_mode=felt, detector=detector)
    overlay = result["overlay"]
    ok, png = cv.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Could not encode analysis overlay")
    best = result["best"]
    response = AnalysisResponse(
        advice=result["advice"],
        outcome=result["outcome"],
        rectified=result["rectified"],
        num_balls=result["num_balls"],
        cue_idx=result["cue_idx"],
        best=Shot(
            score=float(best["score"]),
            object_idx=int(best["object_idx"]),
            pocket=tuple(map(float, best["pocket"])),
            ghost=tuple(map(float, best["ghost"])),
            cut_deg=float(best["cut_deg"]),
        ) if best is not None else None,
        overlay_width=overlay.shape[1],
        overlay_height=overlay.shape[0],
        overlay_png_base64=base64.b64encode(png).decode("ascii"),
    )
    logger.info("%s", json.dumps({
        "event": "analysis_completed",
        "outcome": response.outcome,
        "num_balls": response.num_balls,
        "duration_ms": round((perf_counter() - started) * 1000, 1),
    }))
    return response


@app.post("/analyze", response_model=AnalysisResponse, responses={
    status: {"model": ErrorResponse} for status in (400, 413, 415, 422, 500)
})
async def analyze_endpoint(
    request: Request,
    file: Annotated[UploadFile, File()],
    felt: Literal["auto", "blue", "green"] = "auto",
    detector: Literal["hough"] = "hough",
):
    try:
        return await anyio.to_thread.run_sync(
            analyze_upload, file.file, felt, detector, limiter=request.app.state.analysis_limiter
        )
    except InvalidImage as exc:
        logger.info("%s", json.dumps({"event": "analysis_rejected", "code": exc.code}))
        return error_response(exc.status, exc.code, str(exc))
    except Exception:
        logger.exception('{"event":"analysis_failed"}')
        return error_response(500, "analysis_failed", "The image could not be analyzed.")


def run():
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8080")))


if __name__ == "__main__":
    run()
