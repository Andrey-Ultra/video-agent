"""
Верхний уровень хранилища: всё, что нужно остальному проекту.
Собирает sql_db и chm_db вместе и выбрасывает ошибки вместо None.
Сами запросы к базам живут только в sql_db.py и chm_db.py.
"""
import numpy as np

from . import sql_db
from . import chm_db
from .sql_db import SceneRecord, count_scenes, list_scenes
from video import read_metadata, fast_hash
from ocr.text import contains_score

from models import Scene

MIN_OCR_SCORE = 60.0


# =================
#      ВИДЕО
# =================

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


def get_video_path(video_id: int) -> str:
    path = sql_db.get_video_path_by_id(video_id)
    if path is None:
        raise ValueError(f"Видео {video_id} не найдено")
    return path


# =================
#      СЦЕНЫ
# =================

def add_scene(scene: Scene, vector: np.ndarray) -> int:
    video_id = sql_db.get_video_id_by_path(scene.video_path)
    scene_id = sql_db.add_scene(video_id, scene.start_ms, scene.end_ms)
    chm_db.add_scene_vector(scene_id, vector)

    return scene_id


def get_scene(scene_id: int) -> SceneRecord:
    scene = sql_db.get_scene_by_id(scene_id)
    if scene is None:
        raise ValueError(f"Сцена {scene_id} не найдена")
    return scene


def get_scenes(scene_ids: list[int]) -> list[SceneRecord]:
    return sql_db.get_scenes_by_ids(scene_ids)


def get_scene_neighbors(scene_id: int, n: int = 1) -> tuple[list[SceneRecord], list[SceneRecord]]:
    return sql_db.get_scene_neighbors(get_scene(scene_id), n)


def search_scenes(query_vector: np.ndarray, top_k: int = 10) -> list[tuple[SceneRecord, float]]:
    """Возвращает [(сцена, distance), ...], ближайшие первыми."""
    hits = chm_db.search(query_vector, top_k=top_k)
    distances = dict(hits)
    return [(scene, distances[scene.id]) for scene in get_scenes([scene_id for scene_id, _ in hits])]


# =================
#       OCR
# =================

def add_ocr_spans(video_path: str, spans: list[tuple[int, int, str]]) -> None:
    """Сохраняет распознанный текст видео: [(start_ms, end_ms, text), ...], границы включительны."""
    video_id = sql_db.get_video_id_by_path(video_path)
    if video_id is None:
        raise ValueError(f"Видео не найдено: {video_path}")
    sql_db.add_ocr_spans(video_id, spans)


def search_ocr(query: str, top_k: int = 10) -> list[tuple[sql_db.OcrSpan, float]]:
    """Нечёткий поиск по распознанному тексту на экране. Возвращает [(span, score)], лучшие первыми."""
    scored = [(s, contains_score(query, s.text)) for s in sql_db.get_all_ocr_spans()]
    scored = [x for x in scored if x[1] >= MIN_OCR_SCORE]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def get_scene_ocr_text(scene: SceneRecord) -> list[str]:
    """Тексты из OCR, чьи отрезки пересекаются с диапазоном сцены, в хронологическом порядке."""
    spans = sql_db.get_ocr_spans_in_range(scene.video_id, scene.start_ms, scene.end_ms)
    return [s.text for s in spans]
