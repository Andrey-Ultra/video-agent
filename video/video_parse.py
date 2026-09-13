import logging

import av
from PIL import Image

from models.point import VideoPoint

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, fps_sample: float = 1.0):
    """
    Генератор кадров.
    fps_sample=1.0 -> один кадр в секунду
    fps_sample=0.5 -> один кадр в две секунды
    fps_sample=2.0 -> два кадра в секунду

    Возвращает пары (VideoPoint, PIL.Image)
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"  # ускоряет декодирование

    video_fps = float(stream.average_rate)
    step = int(round(video_fps / fps_sample)) if fps_sample <= video_fps else 1

    for i, frame in enumerate(container.decode(stream)):
        if i % step == 0:
            logger.info("Обработан кадр номер %d", i)

            img = frame.to_image()
            timestamp = float(frame.pts * stream.time_base)

            point = VideoPoint(number=i, timestamp=timestamp)
            yield point, img

    container.close()