import os

import chromadb
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../video-agent/storage
CHROMA_PATH = os.path.join(BASE_DIR, "..", "chroma_data")  # .../video-agent/chroma_data

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_scenes = _client.get_or_create_collection(
    name="scenes_visual",
    metadata={"hnsw:space": "cosine"},
)


def add_scene_vector(scene_id: int, vector: np.ndarray) -> None:
    """Кладёт вектор сцены под её id из SQLite. Повторный вызов обновит вектор."""
    _scenes.upsert(ids=[str(scene_id)], embeddings=[vector.tolist()])


def search(query_vector: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
    """Возвращает [(scene_id, distance), ...], ближайшие первыми."""
    results = _scenes.query(query_embeddings=[query_vector.tolist()], n_results=top_k)
    return [
        (int(scene_id), distance)
        for scene_id, distance in zip(results["ids"][0], results["distances"][0])
    ]


def delete_scene_vectors(scene_ids: list[int]) -> None:
    """Удаляет векторы (например, когда видео пересрезают и старые сцены не нужны)."""
    _scenes.delete(ids=[str(i) for i in scene_ids])