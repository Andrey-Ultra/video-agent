from __future__ import annotations

import av
from PIL import Image

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import Scene

def extract_frames(scene: Scene, fps_sample: float = 1.0):
    """
    Генератор кадров сцены. Возвращает пары (ms, PIL.Image).
    fps_sample: кадров в секунду (1.0 -> раз в секунду, 0.5 -> раз в 2 сек).
    """
    if fps_sample <= 0:
        raise ValueError("fps_sample должен быть > 0")

    step_ms = 1000 / fps_sample

    with av.open(scene.video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        container.seek(scene.start_ms * 1000, stream=None, backward=True)  # микросекунды
        k = 0  # номер следующей точки выборки внутри сцены

        for frame in container.decode(stream):
            t_ms = frame.time * 1000  # секунды -> мс
            if t_ms < scene.start_ms + k * step_ms:
                continue
            if t_ms >= scene.end_ms:
                break

            yield int(t_ms), frame.to_image()
            k = int((t_ms - scene.start_ms) // step_ms) + 1



def extract_n_frames(scene: Scene, n: int):
    """
    Генератор ровно ~n кадров, равномерно по сцене. Возвращает пары (ms, PIL.Image).
    Кадры берутся из середин n равных отрезков, поэтому первый и последний кадры
    сцены (там часто переходы и размытие) не попадают в выборку.
    Если в сцене кадров меньше, чем n, вернётся столько, сколько есть.
    """
    if n < 1:
        raise ValueError("n должно быть >= 1")

    duration = scene.end_ms - scene.start_ms
    targets = [scene.start_ms + (i + 0.5) * duration / n for i in range(n)]

    with av.open(scene.video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        container.seek(scene.start_ms * 1000, stream=None, backward=True)  # микросекунды
        i = 0  # индекс ближайшей целевой точки

        for frame in container.decode(stream):
            t_ms = frame.time * 1000
            if t_ms >= scene.end_ms:
                break
            if t_ms < targets[i]:
                continue

            yield int(t_ms), frame.to_image()

            i += 1
            while i < n and targets[i] <= t_ms:   # короткая сцена: пропускаем уже пройденные точки
                i += 1
            if i >= n:
                break

def extract_frame_at(video_path: str, t_ms: int) -> Image.Image:
    """Один кадр видео на моменте t_ms (первый кадр не раньше этого времени)."""
    t_ms = max(t_ms, 0)
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        container.seek(t_ms * 1000, stream=None, backward=True)

        for frame in container.decode(stream):
            if frame.time * 1000 >= t_ms:
                return frame.to_image()

    raise ValueError(f"В {video_path} нет кадра на {t_ms} мс (время за пределами видео?)")