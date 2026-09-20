import logging
from functools import lru_cache

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from device import pick_device
from models import Scene
from video import extract_n_frames

logger = logging.getLogger(__name__)

MODEL_NAME = "Salesforce/blip-image-captioning-base"


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    device = pick_device()
    logger.info("Загрузка %s (устройство: %s)", model_name, device)
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    return processor, model, device


def describe_scene_fast(scene: Scene, n_frames: int = 3, model_name: str = MODEL_NAME) -> str:
    """Быстрое описание сцены на английском: подписи к нескольким кадрам, склеенные в одну строку."""
    processor, model, device = _load_model(model_name)

    frames = [img.convert("RGB") for _, img in extract_n_frames(scene, n_frames)]
    if not frames:
        raise ValueError(f"В сцене нет кадров: {scene}")

    inputs = processor(images=frames, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    captions = processor.batch_decode(out, skip_special_tokens=True)

    unique = list(dict.fromkeys(c.strip() for c in captions))   # убираем повторы, порядок сохраняем
    text = unique[0].capitalize()
    if len(unique) > 1:
        text += ". Then " + ", then ".join(unique[1:]) + "."

    logger.info("Описана сцена (fast) %s [%d-%d мс]: %s", scene.video_path, scene.start_ms, scene.end_ms, text)
    return text