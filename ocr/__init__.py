import logging

from .easy_ocr import EasyOcrExtractor
from .text import normalize, similarity

from models import Scene
from video import extract_frames

logger = logging.getLogger(__name__)

def scene_to_text(scene: Scene, es: EasyOcrExtractor | None = None):
    BUF_SIZE = 32
    es = es or EasyOcrExtractor()   # лучше передавать готовый: модель грузится долго

    buf = []
    ans = []

    for t, img in extract_frames(scene, fps_sample=0.5):   # 1 кадр в 2 секунды: текст держится дольше
        a = (t, img)

        if len(buf) >= BUF_SIZE:
            ans += es(buf)
            buf = []
            logger.info("OCR: обработано кадров %d (до %.1f с)", len(ans), ans[-1][0] / 1000)

        buf.append(a)

    if len(buf) > 0:
        ans += es(buf)
        buf = []

    logger.info("OCR: всего кадров %d, с текстом %d", len(ans), sum(1 for _, text in ans if text))
    return ans

def scene_to_text_chunks(scene: Scene, es: EasyOcrExtractor | None = None):
    SIM = 95.0
    texts = scene_to_text(scene, es)
    if len(texts) == 0:
        return []

    chunks = []

    seg_start_time = texts[0][0]
    seg_text = texts[0]  # эталонный (текст, время) для сравнения похожести
    last_time = texts[0][0]

    for i in range(1, len(texts)):
        text = texts[i]
        s = similarity(seg_text[1], text[1])

        if s < SIM:
            # текст сменился — закрываем текущий сегмент
            chunks.append((seg_start_time, last_time, seg_text[1]))
            seg_start_time = text[0]
            seg_text = text
        else:
            # тот же сегмент — можно взять более полный вариант текста
            if len(text[1]) > len(seg_text[1]):
                seg_text = text

        last_time = text[0]

    # не забыть закрыть последний сегмент
    chunks.append((seg_start_time, last_time, seg_text[1]))

    for start, end, text in chunks:
        if text:
            logger.info("OCR [%.1f-%.1f с]: %s", start / 1000, end / 1000, text[:80])

    return chunks
