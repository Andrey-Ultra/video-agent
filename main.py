
import storage
from logging_setup import setup_logging
from models import Scene
import logging

from scene_detection import PySceneDetector
from video_embeddings.em_xclip import scene_to_vec

logger = logging.getLogger(__name__)

def main():
    setup_logging()

    folder = "/home/au/Документы/git/video-agent/test"
    scene_detector = PySceneDetector(threshold=40.0)
    storage.init_storage(folder)

    mas = storage.find_videos(folder)

    for video_path in mas:
        if storage.has_video(video_path):
            continue

        storage.create_video(video_path)

        logger.info(f"Обработка видое {video_path}")

        scenes = scene_detector(Scene.from_video(video_path))
        for scene in scenes:
            vec = scene_to_vec(scene)
            storage.add_scene(scene, vec)


if __name__ == "__main__":
    main()