import chromadb
import numpy as np

_scenes = None


def init_vector_store(chroma_path: str) -> None:
    """Открывает (или создаёт) хранилище векторов по указанному пути."""
    global _scenes
    client = chromadb.PersistentClient(path=chroma_path)
    _scenes = client.get_or_create_collection(
        name="scenes_visual",
        metadata={"hnsw:space": "cosine"},
    )


def _collection():
    if _scenes is None:
        raise RuntimeError("Хранилище векторов не инициализировано: вызови init_vector_store()")
    return _scenes


def add_scene_vector(scene_id: int, vector: np.ndarray) -> None:
    """Кладёт вектор сцены под её id из SQLite. Повторный вызов обновит вектор."""
    _collection().upsert(ids=[str(scene_id)], embeddings=[vector.tolist()])


def search(query_vector: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
    """Возвращает [(scene_id, distance), ...], ближайшие первыми."""
    results = _collection().query(query_embeddings=[query_vector.tolist()], n_results=top_k)
    return [
        (int(scene_id), distance)
        for scene_id, distance in zip(results["ids"][0], results["distances"][0])
    ]


def delete_scene_vectors(scene_ids: list[int]) -> None:
    """Удаляет векторы (например, когда видео пересрезают и старые сцены не нужны)."""
    _collection().delete(ids=[str(i) for i in scene_ids])