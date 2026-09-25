import frame_embeddings
from models import Scene
from video import extract_frames

from models import FeatureExtractor, Clusterer

import logging

logger = logging.getLogger(__name__)

MAX_BUF_SIZE = 32


def split_scene(
        scene: Scene,
        embed: FeatureExtractor,
        split: Clusterer,
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



class EmSceneDetector:
    def __init__(self, frame_em: FeatureExtractor, cluster: Clusterer, max_buffer_size: int = MAX_BUF_SIZE):
        self.frame_em = frame_em
        self.cluster = cluster
        self.max_buffer_size = max_buffer_size

    def __call__(self, scene: Scene) -> list[Scene]:
        return split_scene(scene, self.frame_em, self.cluster, self.max_buffer_size)