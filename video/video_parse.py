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