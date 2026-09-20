import logging
from functools import lru_cache

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from models import Scene
from video import extract_n_frames

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"   # для ноутбука с RTX 3050 возьми 2B

PROMPT = (
    "You get {n} frames of one short video scene, in chronological order, "
    "at these times (seconds from the scene start): {times}. "
    "Describe the scene in English in 2-3 sentences: what is shown and what happens over time. "
    "Describe only what is visible, do not invent details. "
    "If there is text on screen, copy it exactly as written, in its original language."
)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Грузит модель один раз за запуск, дальше отдаёт из кэша."""
    device = _pick_device()
    logger.info("Загрузка %s (устройство: %s)", model_name, device)
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16
    ).to(device).eval()
    logger.info("Модель готова")
    return processor, model, device


def describe_scene(
        scene: Scene,
        n_frames: int = 5,
        max_side: int = 640,
        model_name: str = MODEL_NAME,
) -> str:
    """Возвращает описание сцены на английском."""
    processor, model, device = _load_model(model_name)

    frames = list(extract_n_frames(scene, n_frames))
    if not frames:
        raise ValueError(f"В сцене нет кадров: {scene}")

    images = []
    for _, img in frames:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        images.append(img)

    times = ", ".join(f"{(t - scene.start_ms) / 1000:.1f}" for t, _ in frames)
    prompt = PROMPT.format(n=len(frames), times=times)

    messages = [{
        "role": "user",
        "content": [
            *({"type": "image", "image": img} for img in images),
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    logger.info("Описана сцена %s [%d-%d мс]: %s", scene.video_path, scene.start_ms, scene.end_ms, text)
    return text