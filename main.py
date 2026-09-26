
import storage
from logging_setup import setup_logging
from models import Scene
import logging
import os
import time

from scene_detection import PySceneDetector
from video_embeddings.em_xclip import scene_to_vec
from ocr import scene_to_text_chunks, EasyOcrExtractor

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    folder = "/home/au/Документы/git/video-agent/test"
    scene_detector = PySceneDetector(threshold=40.0)
    ocr_extractor = EasyOcrExtractor()
    storage.init_storage(folder)

    mas = storage.find_videos(folder)
    logger.info(f"Найдено видео: {len(mas)} в {folder}")


    for i, video_path in enumerate(mas, 1):
        name = os.path.basename(video_path)
        if storage.has_video(video_path):
            logger.info(f"[{i}/{len(mas)}] {name}: уже в базе, пропускаю")
            continue

        t0 = time.time()

        video_id = storage.create_video(video_path)

        logger.info(f"[{i}/{len(mas)}] Обработка видео {video_path}")

        new_video = Scene.from_video(video_path)

        scenes = scene_detector(new_video)
        for j, scene in enumerate(scenes, 1):
            logger.info(f"Эмбеддинг сцены {j}/{len(scenes)} [{scene.start_ms / 1000:.1f}-{scene.end_ms / 1000:.1f} с]")
            vec = scene_to_vec(scene)
            storage.add_scene(scene, vec)


        # сохранение отрезков из new_video
        logger.info("OCR: распознаю текст")
        chunks = scene_to_text_chunks(new_video, ocr_extractor)
        spans = [c for c in chunks if c[2]]   # кадры без текста не храним
        storage.add_ocr_spans(video_path, spans)
        logger.info(f"OCR: сохранено отрезков с текстом: {len(spans)} (всего отрезков {len(chunks)})")

        logger.info(f"[{i}/{len(mas)}] {name} готово за {time.time() - t0:.0f} с")






if __name__ == "__main__":
    main()