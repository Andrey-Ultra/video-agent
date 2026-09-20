import logging

from scenedetect import ContentDetector, detect

from models import Scene

logger = logging.getLogger(__name__)


def split_by_cuts(scene: Scene, threshold: float = 27.0) -> list[Scene]:
    """
    Режет сцену по склейкам (PySceneDetect, ContentDetector).
    threshold: чем меньше, тем больше склеек находит (по умолчанию 27).
    Если склеек нет, возвращает исходную сцену.
    """
    found = detect(
        scene.video_path,
        ContentDetector(threshold=threshold),
        start_time=scene.start_ms / 1000,
        end_time=scene.end_ms / 1000,
        start_in_scene=True,
    )

    if len(found) <= 1:
        logger.info("Склеек не найдено: %s [%d-%d мс]", scene.video_path, scene.start_ms, scene.end_ms)
        return [scene]

    # берём только начала сцен, конец каждой = начало следующей
    starts = [round(start.seconds * 1000) for start, _ in found]
    starts[0] = scene.start_ms
    ends = starts[1:] + [scene.end_ms]

    result = [
        Scene(start_ms=s, end_ms=e, video_path=scene.video_path)
        for s, e in zip(starts, ends)
        if e > s
    ]

    logger.info("Найдено склеек: %d, сцен: %d", len(result) - 1, len(result))
    return result