import io
import logging
import os

from mcp.server.mcpserver import Image, MCPServer

from export.export_to_shotcut import export_mlt
from logging_setup import setup_logging
from storage import (
    count_video_scenes, find_neighbors, get_scene, get_scene_card, get_scene_cards,
    get_video_path, init_sql_db, list_video_scenes, vector_store,
)
from video import extract_frame_at
from video_embeddings.em_xclip import text_to_vec

from models import Scene
from video import get_duration_ms

logger = logging.getLogger(__name__)

INSTRUCTIONS = """
Video editing assistant. It works with a library of videos that were already cut into scenes.
Every scene has a scene_id, a time range and an English text description.

Typical workflow:
1. Find candidates: search_scenes(query) for scenes, or search_videos(query) to find which videos fit.
   Queries MUST be in English (translate the user's request).
2. Read the descriptions in the results. Use get_frame(video_id, time_s) only when you need to
   actually see a moment.
3. Explore a good video with list_scenes(video_id), or extend a good clip with get_neighbors(scene_id).
4. Assemble the result with export_shotcut(clips, output_path). Each clip is
   {"video_id", "start_ms", "end_ms"}: take video_id and the times from scene cards (converted
   to milliseconds) or use your own times to trim. The order of clips is the order on the timeline.

Notes:
- 'score' is similarity: compare results with each other, it is not an absolute quality.
- Scene cards and get_frame use SECONDS; export_shotcut uses MILLISECONDS (seconds * 1000).
"""

mcp = MCPServer("video-agent", instructions=INSTRUCTIONS)

VIDEO_SEARCH_CANDIDATES = 30   # сколько сцен просматриваем, чтобы определить лучшие видео


def _score(distance: float) -> float:
    return round(1 - distance, 3)


@mcp.tool()
def search_scenes(query: str, top_k: int = 5) -> list[dict]:
    """
    Search video scenes by a text description. The query MUST be in English.
    Returns scene cards, best match first. 'score' is similarity (higher is better);
    compare scores between results rather than treating them as absolute.
    """
    hits = vector_store.search(text_to_vec(query), top_k=min(top_k, 20))
    scores = {scene_id: _score(distance) for scene_id, distance in hits}

    cards = get_scene_cards([scene_id for scene_id, _ in hits])
    return [{**card, "score": scores[card["scene_id"]]} for card in cards]


@mcp.tool()
def search_videos(query: str, top_k: int = 3) -> list[dict]:
    """
    Find the videos that best match a text description (English). Ranked by the best
    matching scene inside each video. Use list_scenes(video_id) to see all its scenes.
    """
    hits = vector_store.search(text_to_vec(query), top_k=VIDEO_SEARCH_CANDIDATES)
    scores = {scene_id: _score(distance) for scene_id, distance in hits}

    videos: dict[int, dict] = {}
    for card in get_scene_cards([scene_id for scene_id, _ in hits]):  # порядок: лучшие первыми
        video = videos.setdefault(card["video_id"], {
            "video_id": card["video_id"],
            "video": card["video"],
            "best_score": scores[card["scene_id"]],
            "matching_scenes": 0,
            "top_scene_ids": [],
        })
        video["matching_scenes"] += 1
        if len(video["top_scene_ids"]) < 3:
            video["top_scene_ids"].append(card["scene_id"])

    ranked = sorted(videos.values(), key=lambda v: v["best_score"], reverse=True)
    return ranked[:top_k]


@mcp.tool()
def list_scenes(video_id: int, offset: int = 0, limit: int = 30) -> dict:
    """
    List the scenes of one video in chronological order (paginated).
    Returns the total scene count and a page of scene cards.
    """
    limit = min(limit, 50)
    return {
        "total": count_video_scenes(video_id),
        "offset": offset,
        "scenes": list_video_scenes(video_id, offset=offset, limit=limit),
    }


@mcp.tool()
def scene_info(scene_id: int) -> dict:
    """Full card of one scene: video, start/end/duration in seconds and description."""
    return get_scene_card(scene_id)


@mcp.tool()
def get_frame(video_id: int, time_s: float) -> Image:
    """Return one frame of a video at the given time in seconds (to visually check a moment)."""
    frame = extract_frame_at(get_video_path(video_id), int(time_s * 1000))
    frame = frame.convert("RGB")
    frame.thumbnail((768, 768))   # экономим токены

    buf = io.BytesIO()
    frame.save(buf, "JPEG", quality=85)
    return Image(data=buf.getvalue(), format="jpeg")


@mcp.tool()
def get_neighbors(scene_id: int, n: int = 1) -> dict:
    """Get the n scenes before and after a scene in the same video (to extend a good clip)."""
    previous, following = find_neighbors(scene_id, n=max(1, min(n, 5)))
    return {"previous": previous, "next": following}


@mcp.tool()
def export_shotcut(clips: list[dict], output_path: str) -> str:
    """
    Export clips, in the given order, as a Shotcut project (.mlt file). Returns the file path.
    Each clip is {"video_id": int, "start_ms": int, "end_ms": int}: any range of a source video,
    in MILLISECONDS (scene cards show seconds, so multiply by 1000). It does not have to match
    scene boundaries.
    """
    if not clips:
        raise ValueError("clips is empty")

    scenes = []
    for clip in clips:
        try:
            video_id, start_ms, end_ms = clip["video_id"], clip["start_ms"], clip["end_ms"]
        except KeyError as e:
            raise ValueError(f"clip is missing {e}: {clip}")

        path = get_video_path(video_id)
        start_ms = max(int(start_ms), 0)
        end_ms = min(int(end_ms), get_duration_ms(path))   # не выходим за конец видео
        if start_ms >= end_ms:
            raise ValueError(f"Empty range: {clip}")

        scenes.append(Scene(start_ms=start_ms, end_ms=end_ms, video_path=path))

    output_path = os.path.abspath(output_path)
    export_mlt(scenes, output_path)
    return output_path


if __name__ == "__main__":
    setup_logging()
    init_sql_db()
    mcp.run(transport="stdio")