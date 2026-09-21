import av
import hashlib
import os

import re
from datetime import datetime

_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


def read_metadata(video_path: str) -> dict:
    with av.open(video_path) as c:
        stream = c.streams.video[0] if c.streams.video else None

        # --- теги: камеры пишут их то в контейнер, то в поток ---
        tags = {}
        if stream is not None:
            tags.update({k.lower(): v for k, v in stream.metadata.items()})
        tags.update({k.lower(): v for k, v in c.metadata.items()})

        # --- длительность ---
        duration_ms = None
        if c.duration is not None:
            duration_ms = c.duration * 1000 // av.time_base
        elif stream is not None and stream.duration is not None:
            duration_ms = int(stream.duration * stream.time_base * 1000)

        # --- размер и fps ---
        width = height = fps = None
        if stream is not None:
            width = stream.codec_context.width
            height = stream.codec_context.height
            rate = stream.average_rate or stream.guessed_rate
            fps = float(rate) if rate else None

            frame = next(c.decode(stream), None)
            if frame is not None and frame.rotation in (90, -90, 270, -270):
                width, height = height, width

    # --- дата съёмки ---
    created_at = None
    if raw := tags.get("creation_time"):
        try:
            created_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass

    # --- GPS: "+55.7558+037.6173+150.000/" ---
    lat = lon = None
    loc = tags.get("com.apple.quicktime.location.iso6709") or tags.get("location")
    if loc and (m := _ISO6709.match(loc)):
        lat, lon = float(m[1]), float(m[2])

    return {
        "duration_ms": duration_ms,
        "width": width,
        "height": height,
        "fps": fps,
        "created_at": created_at,
        "lat": lat,
        "lon": lon,
        "device": tags.get("com.apple.quicktime.model"),
    }


def get_duration_ms(video_path: str) -> int:
    with av.open(video_path) as c:
        if c.duration is not None:
            return c.duration * 1000 // av.time_base
        s = c.streams.video[0]
        return int(s.duration * s.time_base * 1000)


def fast_hash(video_path: str, chunk: int = 4 * 1024 * 1024) -> str:
    size = os.path.getsize(video_path)

    h = hashlib.blake2b(digest_size=16)
    h.update(str(size).encode())

    with open(video_path, "rb") as f:
        for offset in (0, max(0, size // 2 - chunk // 2), max(0, size - chunk)):
            f.seek(offset)
            h.update(f.read(chunk))

    return h.hexdigest()
