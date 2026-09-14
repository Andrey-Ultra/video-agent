import logging
import os
import tempfile

import ffmpeg

logger = logging.getLogger(__name__)

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30


def _get_fps(video_path: str) -> float:
    probe = ffmpeg.probe(video_path)
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    num, denom = stream["r_frame_rate"].split("/")
    return float(num) / float(denom)


def _cut_and_normalize(video_path: str, start_frame: int, end_frame: int, output_path: str):
    """
    Режет кусок по номерам кадров и приводит к единому формату
    (разрешение/fps/кодек), чтобы разные исходники можно было склеить.
    """
    fps = _get_fps(video_path)
    start_time = start_frame / fps
    duration = (end_frame - start_frame + 1) / fps

    (
        ffmpeg
        .input(video_path, ss=start_time)
        .output(
            output_path,
            t=duration,
            vf=f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
               f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
            r=TARGET_FPS,
            vcodec="libx264",
            acodec="aac",
            ar=44100,
        )
        .overwrite_output()
        .run(quiet=True)
    )


def assemble_video(segments: list[tuple[str, int, int]], output_path: str):
    """
    segments: [(video_path, start_frame, end_frame), ...]
    Каждый сегмент может быть из своего исходника, любого формата.
    Режет, нормализует, склеивает по порядку в output_path.
    """
    if not segments:
        raise ValueError("Пустой список сегментов — нечего склеивать")

    with tempfile.TemporaryDirectory() as tmp_dir:
        chunk_paths = []

        for i, (video_path, start, end) in enumerate(segments):
            chunk_path = os.path.join(tmp_dir, f"chunk_{i}.mp4")
            _cut_and_normalize(video_path, start, end, chunk_path)
            chunk_paths.append(chunk_path)
            logger.info("Нарезан и нормализован кусок %d/%d (%s, %d-%d)", i + 1, len(segments), video_path, start, end)

        concat_list_path = os.path.join(tmp_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for path in chunk_paths:
                f.write(f"file '{path}'\n")

        (
            ffmpeg
            .input(concat_list_path, format="concat", safe=0)
            .output(output_path, c="copy")
            .overwrite_output()
            .run(quiet=True)
        )

        logger.info("Итоговое видео сохранено: %s", output_path)

    return output_path