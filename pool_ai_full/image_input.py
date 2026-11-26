from io import BytesIO

import cv2 as cv
import numpy as np
from PIL import Image

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


class InvalidImage(ValueError):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code


def invalid_image():
    return InvalidImage(400, "bad_image", "Could not decode the image.")


def too_many_pixels():
    return InvalidImage(413, "image_too_large", "Image exceeds 16 million decoded pixels.")


def decode_image(data):
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidImage(413, "image_too_large", "Image exceeds 8 MiB.")
    if not data:
        raise invalid_image()
    if not data.startswith((b"\xff\xd8", b"\x89PNG\r\n\x1a\n")):
        raise InvalidImage(415, "unsupported_image", "Upload a JPEG or PNG image.")
    try:
        with Image.open(BytesIO(data), formats=("JPEG", "PNG")) as header:
            width, height = header.size
    except Image.DecompressionBombError:
        raise too_many_pixels() from None
    except (OSError, SyntaxError, ValueError):
        raise invalid_image() from None
    if width * height > MAX_IMAGE_PIXELS:
        raise too_many_pixels()
    try:
        image = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    except cv.error:
        raise invalid_image() from None
    if image is None:
        raise invalid_image()
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise too_many_pixels()
    return image
