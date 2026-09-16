import av
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def extract_frames(video_path: str, fps_sample: float = 1.0):
    """
    Генератор кадров.
    fps_sample=1.0 -> один кадр в секунду
    fps_sample=0.5 -> один кадр в две секунды
    fps_sample=2.0 -> два кадра в секунду

    Возвращает пары (n, PIL.Image) номер кадра и кадр
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"  # ускоряет декодирование

    video_fps = float(stream.average_rate)
    step = int(round(video_fps / fps_sample)) if fps_sample <= video_fps else 1

    logger.info(f"Извлечение кадров из {video_path}")
    for i, frame in enumerate(container.decode(stream)):
        if i % step == 0:
            logger.info(f"Извлечён кадр {i}")
            img = frame.to_image()
            yield i, img

    container.close()

def get_video_info(video_path: str) -> tuple[int, float]:
    """Возвращает (количество кадров, средний FPS)."""
    with av.open(video_path) as container:
        stream = container.streams.video[0]

        if stream.average_rate is None or stream.average_rate <= 0:
            raise ValueError("Не удалось определить FPS видео")

        fps = float(stream.average_rate)
        frame_count = stream.frames

        if frame_count <= 0:
            frame_count = sum(1 for _ in container.decode(stream))

        return frame_count, fps