import chromadb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../video-agent/storage
CHROMA_PATH = os.path.join(BASE_DIR, "..", "chroma_data")  # .../video-agent/chroma_data

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name="video_segments",
    metadata={"hnsw:space": "cosine"}
)


def add_segment(segment_id, vector, video_path, start_frame, end_frame, start_ts, end_ts, thumbnail_path):
    _collection.add(
        ids=[segment_id],
        embeddings=[vector.tolist()],
        metadatas=[{
            "video_path": video_path,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "thumbnail_path": thumbnail_path,
        }]
    )

def search(query_vector, top_k: int = 10, where: dict | None = None):
    results = _collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        where=where,
    )
    return results


def get_metadata(segment_id: str) -> dict:
    result = _collection.get(ids=[segment_id])
    if not result["ids"]:
        raise ValueError(f"Сегмент {segment_id} не найден в БД")
    return result["metadatas"][0]