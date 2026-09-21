import os

import numpy as np

from . import sql_db
from . import chm_db
from video import read_metadata, fast_hash

from models import Scene


def has_video(video_path: str) -> int | None:
    hash = fast_hash(video_path)
    id = sql_db.get_video_id_by_hash(hash)

    if id:
        sql_db.update_video_path(id, video_path)
        return id

    return None


def create_video(video_path: str) -> int:
    hash = fast_hash(video_path)
    metadata = read_metadata(video_path)

    return sql_db.add_video(hash, video_path, metadata)



def add_scene(scene:Scene, vector: np.ndarray):
    video_id = sql_db.get_video_id_by_path(scene.video_path)
    scene_id = sql_db.add_scene(video_id, scene.start_ms, scene.end_ms)
    chm_db.add_scene_vector(scene_id, vector)

    return scene_id