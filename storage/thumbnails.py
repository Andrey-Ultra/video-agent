import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../video-agent/storage
THUMBNAILS_DIR = os.path.join(BASE_DIR, "thumbnails_data")
os.makedirs(THUMBNAILS_DIR, exist_ok=True)


def save_thumbnail(buffer: list[Image.Image], segment_id: str) -> str:
    mid_frame = buffer[len(buffer) // 2]
    path = os.path.join(THUMBNAILS_DIR, f"{segment_id}.jpg")
    mid_frame.convert("RGB").save(path, "JPEG", quality=85)
    return path