import base64
import logging

from mcp.server.mcpserver import Image
from mcp.server.mcpserver import MCPServer

from storage import vector_store
from video.assembler import assemble_video
from video_embeddings import em_xclip  # для эмбеддинга текстового запроса

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = MCPServer("video-agent")



@mcp.tool()
def search_clips(query: str, top_k: int = 10) -> list[dict]:
    """
    Ищет видео-сегменты по текстовому запросу (семантический поиск).
    Возвращает список сегментов с метаданными: путь к видео, таймкоды, id.
    """
    query_vec = em_xclip.text_to_vec(query)  # нужно добавить эту функцию в em_xclip
    results = vector_store.search(query_vector=query_vec, top_k=top_k)

    segments = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        segments.append({
            "segment_id": results["ids"][0][i],
            "video_path": meta["video_path"],
            "start_frame": meta["start_frame"],
            "end_frame": meta["end_frame"],
            "start_ts": meta.get("start_ts"),
            "end_ts": meta.get("end_ts"),
            "distance": results["distances"][0][i],
        })
    return segments



@mcp.tool()
def get_clip_preview(segment_id: str) -> Image:
    """
    Возвращает миниатюру (превью-кадр) сегмента, чтобы агент мог визуально оценить клип.
    """
    meta = vector_store.get_metadata(segment_id)
    return Image(path=meta["thumbnail_path"])


@mcp.tool()
def render_video(segments: list[dict], output_path: str) -> str:
    """
    Склеивает переданные сегменты в один файл по порядку.
    segments: [{"video_path": ..., "start_frame": ..., "end_frame": ...}, ...]
    """
    tuples = [(s["video_path"], s["start_frame"], s["end_frame"]) for s in segments]
    result_path = assemble_video(tuples, output_path)
    return result_path


if __name__ == "__main__":
    mcp.run(transport="stdio")