"""Bounded image decoding shared by every HTTP image endpoint."""
from io import BytesIO
import warnings

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000
FORMATS = {'image/jpeg': ('JPEG', '.jpg'), 'image/png': ('PNG', '.png'),
           'image/webp': ('WEBP', '.webp')}


def decode_image(raw, content_type):
    expected = FORMATS.get((content_type or '').lower())
    if expected is None:
        raise HTTPException(415, 'Only JPEG, PNG, and WebP images are supported')
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, 'Image must be 10 MB or smaller')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as header:
                if header.width * header.height > MAX_PIXELS:
                    raise HTTPException(413, 'Image exceeds the 24 megapixel limit')
                if header.format != expected[0]:
                    raise HTTPException(415, 'Image bytes do not match the content type')
                header.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(413, 'Image dimensions are too large')
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(400, 'The uploaded file is not a valid image')
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, 'The uploaded file is not a valid image')
    return image, expected[1]
