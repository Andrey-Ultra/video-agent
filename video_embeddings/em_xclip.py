from functools import lru_cache

import numpy as np
import torch
from transformers import XCLIPModel, XCLIPProcessor

from device import pick_device
from models import Scene
from video import extract_n_frames

MODEL_NAME = "microsoft/xclip-base-patch32"
NUM_FRAMES = 8  # столько кадров ждёт этот чекпоинт X-CLIP


@lru_cache(maxsize=1)
def _load(model_name: str = MODEL_NAME):
    """Грузит модель один раз за запуск, дальше отдаёт из кэша."""
    device = pick_device()
    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name).to(device).eval()
    return processor, model, device


def _as_vector(output, attr: str) -> np.ndarray:
    """Достаёт тензор из ответа модели (у разных версий transformers он лежит по-разному)."""
    if hasattr(output, "pooler_output"):
        embedding = output.pooler_output
    elif hasattr(output, attr):
        embedding = getattr(output, attr)
    else:
        embedding = output
    return embedding.squeeze().cpu().detach().numpy()


def scene_to_vec(scene: Scene) -> np.ndarray:
    """Вектор сцены: 8 кадров, равномерно по времени."""
    processor, model, device = _load()

    frames = [img for _, img in extract_n_frames(scene, NUM_FRAMES)]
    if not frames:
        raise ValueError(f"В сцене нет кадров: {scene}")

    # если кадров вышло меньше 8 (очень короткая сцена), часть повторится
    idx = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
    sampled = [frames[i] for i in idx]

    inputs = processor(videos=[sampled], return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.get_video_features(**inputs)

    return _as_vector(output, "video_embeds")


def text_to_vec(text: str) -> np.ndarray:
    """Вектор текстового запроса (из того же пространства, что и сцены)."""
    processor, model, device = _load()

    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        output = model.get_text_features(**inputs)

    return _as_vector(output, "text_embeds")