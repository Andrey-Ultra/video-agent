import av
from PIL import Image

def extract_frames(video_path, fps_sample=1.0):
    """
    fps_sample=1.0 -> один кадр в секунду
    fps_sample=0.5 -> один кадр в две секунду
    fps_sample=2.0 -> два кадра в секунду
    Возвращает список (timestamp_sec, PIL.Image)
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"  # ускоряет декодирование

    video_fps = float(stream.average_rate)
    step = int(round(video_fps / fps_sample)) if fps_sample <= video_fps else 1

    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i % step == 0:
            img = frame.to_image()  # сразу PIL.Image, без ручной конвертации
            timestamp = float(frame.pts * stream.time_base)
            frames.append((timestamp, img))

    container.close()
    return frames


def test():
    video_path = "/Users/au/Documents/git/video-agent/frames/video.MOV"
    frames = extract_frames(video_path)

    for i in frames:
        img = i[1]
        name = i[0]
        img.save(f"/Users/au/Documents/git/video-agent/frames/{name}.png")



if __name__ == "__main__":
    test()