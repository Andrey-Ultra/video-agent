from models import Scene
from video import extract_frames

from models.interfaces import Embedder, Splitter

import logging

logger = logging.getLogger(__name__)

MAX_BUF_SIZE = 32


def split_scene(
        scene: Scene,
        embed: Embedder,
        split: Splitter,
        max_buffer_size: int = MAX_BUF_SIZE
) -> list[Scene]:
    buf_frame = []
    buf_vec = []
    video_path = scene.video_path

    for t_ms, img in extract_frames(scene):

        logger.info(f"Processing frame {t_ms / 1000:.2f} s")

        buf_frame.append((t_ms, img))

        if len(buf_frame) >= max_buffer_size:
            buf_vec += embed(buf_frame)
            buf_frame.clear()

    if buf_frame:
        buf_vec += embed(buf_frame)
        buf_frame.clear()

    ans = split(buf_vec)
    if len(ans) <= 1:
        return [scene]

    scenes_ans = []
    ind = 0

    logger.info(f"Processing {len(ans)} scenes")

    for s, e in ans:
        if ind == 0:
            s = scene.start_ms
        if ind == len(ans) - 1:
            e = scene.end_ms

        scenes_ans.append(Scene(start_ms=s, end_ms=e, video_path=video_path))
        ind += 1

    return scenes_ans
