import os

from .sql_db import init_sql_db
from .chm_db import init_vector_store
from .sql_db import SceneRecord
from .library import (
    has_video, create_video, get_video_path,
    add_scene, get_scene, get_scenes, get_scene_neighbors, search_scenes,
    count_scenes, list_scenes,
    add_ocr_spans,
)

LIB_DIR = ".video-agent"


def init_storage(folder: str) -> str:
    """
    Подключает хранилище к папке с видео.
    Если в ней нет .video-agent — создаёт, если есть — открывает существующее.
    Возвращает абсолютный путь к папке (корень библиотеки).
    """
    root = os.path.abspath(folder)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Папка не найдена: {root}")

    lib = os.path.join(root, LIB_DIR)
    os.makedirs(lib, exist_ok=True)

    init_sql_db(os.path.join(lib, "index.db"))
    init_vector_store(os.path.join(lib, "chroma"))
    return root


VIDEO_EXTS = {
    # телефоны и большинство камер
    ".mp4", ".mov", ".m4v",
    # запись геймплея: OBS по умолчанию пишет в .mkv
    ".mkv", ".webm", ".flv",
    # видеокамеры (AVCHD) и транспортные потоки
    ".mts", ".m2ts", ".ts",
    # старые и прочие форматы
    ".avi", ".wmv", ".mpg", ".mpeg", ".3gp",
    # профессиональные камеры
    ".mxf",
}


def find_videos(folder: str) -> list[str]:
    """Рекурсивно находит все видео в папке. Возвращает отсортированный список путей."""
    result = []

    for dirpath, dirnames, filenames in os.walk(folder):
        # не заходить в скрытые папки, в том числе в свою .video-agent
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for name in filenames:
            if name.startswith("."):
                continue
            if os.path.splitext(name)[1].lower() in VIDEO_EXTS:
                result.append(os.path.join(dirpath, name))

    return sorted(result)