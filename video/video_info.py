import av

def get_duration_ms(video_path: str) -> int:
    with av.open(video_path) as c:
        if c.duration is not None:
            return c.duration * 1000 // av.time_base   # container.duration в микросекундах
        s = c.streams.video[0]
        return int(s.duration * s.time_base * 1000)