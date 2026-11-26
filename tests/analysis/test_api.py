import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pool_ai_full.detectors import HoughBallDetector
from pool_ai_full.server import app


SAMPLE = Path(__file__).resolve().parents[2] / "sample_pool_table1.jpg"


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def png_bytes(image):
    encoded, png = cv2.imencode(".png", image)
    assert encoded
    return png.tobytes()


def upload(client, content, filename="table.png", media_type="image/png", **params):
    return client.post(
        "/analyze",
        params=params,
        files={"file": (filename, content, media_type)},
    )


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["code"] == code
    assert error["message"]


def decode_overlay(result):
    png = base64.b64decode(result["overlay_png_base64"], validate=True)
    overlay = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert overlay is not None
    assert overlay.shape[:2] == (result["overlay_height"], result["overlay_width"])
    return overlay


@pytest.mark.parametrize("endpoint", ["/health", "/ready"])
def test_health_endpoints(client, endpoint):
    assert client.get(endpoint).status_code == 200


def test_readiness_reports_unavailable_encoder(client, monkeypatch):
    monkeypatch.setattr(cv2, "haveImageWriter", lambda _extension: False)
    assert_error(client.get("/ready"), 503, "not_ready")
    assert client.get("/health").status_code == 200


def test_sample_produces_json_safe_shot_and_overlay(client):
    response = upload(client, SAMPLE.read_bytes(), SAMPLE.name, "image/jpeg", felt="auto")
    assert response.status_code == 200, response.text
    result = response.json()

    assert result["outcome"] == "recommendation"
    assert result["rectified"] is True
    assert result["num_balls"] >= 2
    assert 0 <= result["cue_idx"] < result["num_balls"]
    shot = result["best"]
    assert shot["object_idx"] != result["cue_idx"]
    assert 0 <= shot["object_idx"] < result["num_balls"]
    assert 0 <= shot["cut_deg"] <= 80
    assert isinstance(shot["score"], (int, float))
    for point in (shot["pocket"], shot["ghost"]):
        assert len(point) == 2
        assert all(isinstance(value, (int, float)) for value in point)
    assert result["advice"]
    assert decode_overlay(result).shape == (512, 1024, 3)


def test_blank_image_returns_no_usable_table_and_original_dimensions(client):
    content = png_bytes(np.zeros((96, 144, 3), dtype=np.uint8))
    response = upload(client, content, felt="green")
    assert response.status_code == 200, response.text
    result = response.json()

    assert result["outcome"] == "no_usable_table"
    assert result["rectified"] is False
    assert result["best"] is None
    assert result["advice"]
    assert decode_overlay(result).shape[:2] == (96, 144)


def test_rectified_empty_table_returns_no_cue_ball(client):
    image = np.zeros((280, 540, 3), dtype=np.uint8)
    image[12:-12, 12:-12] = (0, 180, 0)
    response = upload(client, png_bytes(image), felt="green")
    assert response.status_code == 200, response.text
    result = response.json()

    assert result["outcome"] == "no_cue_ball"
    assert result["rectified"] is True
    assert result["num_balls"] == 0
    assert result["cue_idx"] is None
    assert result["best"] is None


def test_unrectified_image_skips_ball_detection(client, monkeypatch):
    def detect(_detector, _image):
        pytest.fail("Hough detection should not run on an unrectified image")

    monkeypatch.setattr(HoughBallDetector, "detect", detect)
    image = np.zeros((300, 600, 3), dtype=np.uint8)
    response = upload(client, png_bytes(image), felt="green")
    assert response.status_code == 200, response.text
    result = response.json()

    assert result["num_balls"] == 0
    assert result["cue_idx"] is None
    assert result["outcome"] == "no_usable_table"
    assert result["best"] is None


def test_cue_ball_without_target_returns_no_clear_pot(client, monkeypatch):
    def detect(_detector, _image):
        return [(512, 250, 10)], 0

    monkeypatch.setattr(HoughBallDetector, "detect", detect)
    image = np.zeros((280, 540, 3), dtype=np.uint8)
    image[12:-12, 12:-12] = (0, 180, 0)
    response = upload(client, png_bytes(image), felt="green")
    assert response.status_code == 200, response.text
    result = response.json()

    assert result["outcome"] == "no_clear_pot"
    assert result["cue_idx"] == 0
    assert result["best"] is None


def test_upload_errors(client):
    assert_error(upload(client, b""), 400, "bad_image")
    assert_error(upload(client, b"plain text"), 415, "unsupported_image")
    assert_error(
        upload(client, b"\x89PNG\r\n\x1a\n"),
        400,
        "bad_image",
    )
    assert_error(client.post("/analyze"), 422, "validation_error")
    assert_error(upload(client, b"plain text", felt="red"), 422, "validation_error")


def test_malformed_multipart_returns_upload_error(client):
    response = client.post(
        "/analyze",
        headers={"Content-Type": "multipart/form-data"},
        content=b"missing boundary",
    )
    assert_error(response, 400, "invalid_upload")


def test_encoded_upload_limit(client):
    content = b"x" * (8 * 1024 * 1024 + 1)
    assert_error(upload(client, content), 413, "image_too_large")


def test_streamed_multipart_is_bounded_without_content_length(client):
    def chunks():
        yield (
            b"--test-boundary\r\n"
            b'Content-Disposition: form-data; name="file"; filename="table.png"\r\n'
            b"Content-Type: image/png\r\n\r\n"
        )
        for _ in range(9):
            yield b"x" * (1024 * 1024)
        yield b"\r\n--test-boundary--\r\n"

    response = client.post(
        "/analyze",
        headers={"Content-Type": "multipart/form-data; boundary=test-boundary"},
        content=chunks(),
    )
    assert "content-length" not in response.request.headers
    assert_error(response, 413, "image_too_large")


def test_decoded_image_limit(client):
    image = np.zeros((4001, 4000), dtype=np.uint8)
    assert_error(upload(client, png_bytes(image)), 413, "image_too_large")
