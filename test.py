
import storage
from logging_setup import setup_logging
from models import Scene
from scene_detection.cut_detection import split_by_cuts
from video_embeddings.em_xclip import scene_to_vec
import logging

logger = logging.getLogger(__name__)

def main():
    setup_logging()

    folder = "/Users/au/Documents/git/video-agent/test"

    storage.init_storage(folder)

    mas = storage.find_videos(folder)

    for video_path in mas:
        if storage.has_video(video_path):
            continue

        storage.create_video(video_path)

        logger.info(f"Обработка видое {video_path}")

        scenes = split_by_cuts(Scene.from_video(video_path))
        for scene in scenes:
            vec = scene_to_vec(scene)
            storage.add_scene(scene, vec)


if __name__ == "__main__":
    main()