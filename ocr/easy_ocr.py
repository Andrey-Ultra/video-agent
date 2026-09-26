import logging

import numpy as np
from easyocr import Reader

from device import pick_device
from models import Frame
from .text import normalize

logger = logging.getLogger(__name__)


class EasyOcrExtractor:
    """Извлечение текста из кадров: (мс, картинка) -> (мс, текст)."""

    def __init__(
            self,
            languages: list[str] = ["ru", "en"],
            device: str | None = None,
            canvas_size: int = 1280,   # у EasyOCR по умолчанию 2560 — на CPU это очень медленно
    ):
        self.device = device or pick_device()
        self.canvas_size = canvas_size

        logger.info("Загрузка EasyOCR: %s, устройство: %s", languages, self.device)
        # gpu принимает bool или строку с именем устройства ('cuda' / 'mps' / 'cpu')
        self._reader = Reader(languages, gpu=self.device)
        logger.info("EasyOCR готов")

    def __call__(self, frames: list[Frame]) -> list[tuple[int, str]]:
        results: list[tuple[int, str]] = []

        for t, img in frames:
            arr = np.array(img)
            lines = self._reader.readtext(arr, detail=0, canvas_size=self.canvas_size)
            results.append((t, normalize(" ".join(lines))))

        return results