import logging

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from models.embedding import ClipEmbedding
from models.interfaces import Frame, Vec

logger = logging.getLogger(__name__)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipEmbedder:
    """Реализация Embedder: кадры (мс, картинка) -> признаки (мс, ClipEmbedding)."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.batch_size = batch_size
        self.device = device or _pick_device()

        logger.info("Загрузка CLIP: %s", model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        logger.info("CLIP готов, устройство: %s", self.device)

    def __call__(self, frames: list[Frame]) -> list[Vec]:
        results: list[Vec] = []

        for i in range(0, len(frames), self.batch_size):
            chunk = frames[i:i + self.batch_size]
            times = [t for t, _ in chunk]
            images = [img for _, img in chunk]

            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self._model.get_image_features(**inputs)

            if hasattr(output, "pooler_output"):
                embeds = output.pooler_output
            elif hasattr(output, "image_embeds"):
                embeds = output.image_embeds
            else:
                embeds = output

            embeds = embeds.cpu().numpy()
            results.extend((t, ClipEmbedding(v)) for t, v in zip(times, embeds))

        return results