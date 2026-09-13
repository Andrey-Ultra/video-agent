import ffmpeg
import subprocess


def get_video_fps(video_path: str) -> float:
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    num, denom = video_stream["r_frame_rate"].split("/")
    return float(num) / float(denom)


def cut_video_by_frames(video_path: str, start_frame: int, end_frame: int, output_path: str):
    """
    Универсально режет видео любого формата по номерам кадров и сохраняет как mp4.
    """
    fps = get_video_fps(video_path)
    start_time = start_frame / fps
    duration = (end_frame - start_frame + 1) / fps

    (
        ffmpeg
        .input(video_path, ss=start_time)
        .output(output_path, t=duration, vcodec="libx264", acodec="aac")
        .overwrite_output()
        .run(quiet=True)
    )