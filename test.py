from logging_setup import setup_logging
from models import Scene
from scene_detection import PySceneDetector
from export import MLTExporter
import logging

logger = logging.getLogger(__name__)

def main():
    setup_logging()

    video_path = "/home/au/Документы/git/video-agent/test/park.MOV"

    scene_detector = PySceneDetector(threshold=40.0)
    export = MLTExporter()


    scenes = scene_detector(Scene.from_video(video_path))
    export(scenes, "/home/au/Документы/git/video-agent/test/some.mlt")


if __name__ == "__main__":
    main()