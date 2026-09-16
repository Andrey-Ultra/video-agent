import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

logger = logging.getLogger(__name__)
logger.info("Запуск приложения")
from scene_detection.embedding_detection import split_scenes
from models import Scene

def main():

    video = Scene.from_video("/home/au/Документы/git/video-agent/test/park.MOV")

    ans = split_scenes([video])

    print(ans)


if __name__ == '__main__':

    main()