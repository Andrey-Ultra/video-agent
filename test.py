from logging_setup import setup_logging
from models import Scene
from scene_detection import PySceneDetector
from export import MLTExporter
from video import extract_frames
from models import Frame
from ocr import scene_to_text_chunks
import logging

logger = logging.getLogger(__name__)

def main():
    setup_logging()

    video_path = "/home/au/Документы/git/video-agent/test/screen.mp4"
    scene = Scene.from_video(video_path)

    print(scene_to_text_chunks(scene))



if __name__ == "__main__":
    main()