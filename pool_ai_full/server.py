#!/usr/bin/env python3
import io, os, base64
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import uvicorn, cv2 as cv, numpy as np
from .analyze import analyze_pool_image

app = FastAPI()

@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    felt: str = Query("auto", regex="^(auto|green|blue)$"),
    detector: str = Query("hough", regex="^(hough)$")
):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error":"bad image"}, status_code=400)

    res = analyze_pool_image(img, felt_mode=felt, detector=detector)
    ok, png = cv.imencode(".png", res["overlay"])
    overlay_b64 = base64.b64encode(png.tobytes()).decode("ascii") if ok else None

    return {
        "advice": res["advice"],
        "rectified": res["rectified"],
        "num_balls": res["num_balls"],
        "cue_idx": res["cue_idx"],
        "best": res["best"],
        "overlay_png_base64": overlay_b64
    }

def run():
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

if __name__ == "__main__":
    run()
