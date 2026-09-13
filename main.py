from video import video_parse
from frame_embeddings import em_clip
from scene_detection import simple_em
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    video_path = "/Users/au/Documents/git/video-agent/test/IMG_3602 2.MOV"

    mas = []
    last_vec = None
    ms = []

    for point, frame in video_parse.extract_frames(video_path, fps_sample=1.0):
        vec = em_clip.frame_to_vec(frame)
        logger.info(f"Обработан {frame}")

        mas.append((point, vec))

        if last_vec:
            ms.append(vec.distance(last_vec))

        last_vec = vec


    print(ms)

    scenes = simple_em.simple(mas, 0.2)
    logger.info("Найдено сцен: %d", len(scenes))

    print(scenes)


if __name__ == "__main__":
    main()
