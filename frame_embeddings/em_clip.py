import logging

import torch
from transformers import CLIPModel, CLIPProcessor
from device import pick_device

from frame_embeddings import EmVector
from models import Frame, Vec

logger = logging.getLogger(__name__)


class ClipEmbedder:
    """Реализация Embedder: кадры (мс, картинка) -> признаки (мс, ClipEmbedding)."""

    def __init__(
            self,
            model_name: str = "openai/clip-vit-large-patch14",
            batch_size: int = 32,
            device: str | None = None,
    ):
        self.batch_size = batch_size
        self.device = device or pick_device()

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
            results.extend((t, EmVector(v)) for t, v in zip(times, embeds))

        return results
