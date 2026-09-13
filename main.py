from video import video_parse
from frame_embeddings import em_clip
import logging
import slice
import cutter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    video_path = "/test/park.MOV"

    mas = []
    for frame in video_parse.extract_frame(video_path, fps_sample=1.0):
        vec = em_clip.frame_to_vec(frame)

        mas.append(vec)

    a = slice.simple(mas)



if __name__ == "__main__":
    main()