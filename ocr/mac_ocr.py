import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import Quartz
import Vision
from Foundation import NSData
from PIL import Image

from models import Frame
from .text import normalize

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class VisionOcrExtractor:
    def __init__(
            self,
            languages: list[str] = ["ru-RU", "en-US"],
            max_side: int = 1000,
            max_workers: int = 4,
    ):
        self.languages = languages
        self.max_side = max_side
        self.max_workers = max_workers
        self._warmup()

    def _warmup(self):
        """Прогоняем один крошечный кадр в главном потоке, чтобы pyobjc
        успел лениво резолвнуть все нужные символы ДО того, как несколько
        потоков полезут за ними одновременно."""
        dummy = Image.new("RGB", (10, 10), color="white")
        self._process(dummy)

    def _resize(self, img):
        w, h = img.size
        if max(w, h) <= self.max_side:
            return img
        scale = self.max_side / max(w, h)
        return img.resize((int(w * scale), int(h * scale)))

    def _process(self, img) -> str:
        img = self._resize(img)

        # Всё, что трогает "ленивые" символы Quartz/Vision, защищаем локом —
        # это подстраховка на случай, если warmup что-то не покрыл.
        with _lock:
            buf = BytesIO()
            img.save(buf, format="PNG")
            data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
            provider = Quartz.CGDataProviderCreateWithCFData(data)
            cgimage = Quartz.CGImageCreateWithPNGDataProvider(
                provider, None, True, Quartz.kCGRenderingIntentDefault
            )

            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRevision_(Vision.VNRecognizeTextRequestRevision3)
            request.setRecognitionLevel_(0)  # accurate — только он понимает ru-RU
            request.setRecognitionLanguages_(self.languages)
            request.setUsesLanguageCorrection_(False)

            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)

        # Сам OCR (самая тяжёлая часть) — уже без лока, параллелится нормально
        handler.performRequests_error_([request], None)

        return " ".join(
            obs.topCandidates_(1)[0].string()
            for obs in (request.results() or [])
        )

    def __call__(self, frames: list[Frame]) -> list[tuple[int, str]]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            texts = list(pool.map(lambda f: self._process(f[1]), frames))

        return [(t, normalize(txt)) for (t, _), txt in zip(frames, texts)]