import os

from video import video_parse
from frame_embeddings import em_clip
from scene_detection import adaptive_em
from video_embeddings import em_xclip
import uuid
from storage import chm_db, thumbnails
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FPS_SAMPLE = 10.0

def main():
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test", "/home/au/Документы/git/video-agent/test/scot-2.mp4")

    mas = []
    last_vec = None
    ms = []

    for point, frame in video_parse.extract_frames(video_path, fps_sample=FPS_SAMPLE):
        vec = em_clip.frame_to_vec(frame)
        logger.info(f"Обработан {frame}")

        mas.append((point, vec))

        if last_vec:
            ms.append(vec.distance(last_vec))

        last_vec = vec


    print(ms)

    scenes = adaptive_em.adaptive(mas, 2)
    logger.info("Найдено сцен: %d", len(scenes))

    print(scenes)

    # --- второй проход: эмбеддинг видео-сегментов ---
    seg_idx = 0
    buffer = []

    seg_start_ts = 0

    for point, frame in video_parse.extract_frames(video_path, fps_sample=FPS_SAMPLE):
        seg_start, seg_end = scenes[seg_idx]

        buffer.append(frame)

        if point.number >= seg_end:
            segment_id = str(uuid.uuid4())

            vec = em_xclip.segment_to_vec(buffer)
            thumb_path = thumbnails.save_thumbnail(buffer, segment_id)

            chm_db.add_segment(
                segment_id=segment_id,
                vector=vec,
                video_path=video_path,
                start_frame=seg_start,
                end_frame=seg_end,
                start_ts=seg_start_ts,
                end_ts=point.timestamp,
                thumbnail_path=thumb_path,
            )

            logger.info("Сегмент %d (%d-%d) сохранён в БД, миниатюра: %s", seg_idx, seg_start, seg_end, thumb_path)

            buffer = []
            seg_start_ts = point.timestamp
            seg_idx += 1

            if seg_idx >= len(scenes):
                break



if __name__ == "__main__":
    main()
