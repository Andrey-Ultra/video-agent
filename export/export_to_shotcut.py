import os
import xml.etree.ElementTree as ET

import av

from models import Scene
from video import get_duration_ms


def _ts(ms: int) -> str:
    """Миллисекунды -> 'ЧЧ:ММ:СС.ммм'."""
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def _stream_indices(path: str) -> tuple[int, int]:
    """(индекс видеопотока, индекс аудиопотока), -1 если потока нет."""
    with av.open(path) as c:
        v = next((s.index for s in c.streams if s.type == "video"), -1)
        a = next((s.index for s in c.streams if s.type == "audio"), -1)
    return v, a


def _add_chain(root, chain_id: str, path: str, full_ms: int, *, video: bool) -> None:
    """chain на весь файл. video=True -> только картинка, False -> только звук."""
    v_idx, a_idx = _stream_indices(path)

    chain = ET.SubElement(root, "chain", id=chain_id, out=_ts(full_ms - 1))
    props = [
        ("length", _ts(full_ms)),
        ("eof", "pause"),
        ("resource", path),
        ("mlt_service", "avformat-novalidate"),
        ("seekable", "1"),
        ("shotcut:caption", os.path.basename(path)),
    ]
    if video:
        props += [("video_index", str(v_idx)), ("audio_index", "-1"), ("astream", "-1")]
    else:
        props += [("audio_index", str(a_idx)), ("video_index", "-1"), ("vstream", "-1")]
    for name, value in props:
        ET.SubElement(chain, "property", name=name).text = value


def export_mlt(
        scenes: list[Scene],
        out_path: str,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
) -> None:
    """
    Сохраняет сцены в MLT-проект (Shotcut). Порядок списка = порядок на таймлайне.
    """
    if not scenes:
        raise ValueError("Нет сцен для экспорта")

    total_ms = sum(s.end_ms - s.start_ms for s in scenes)
    t0, t_end = _ts(0), _ts(total_ms - 1)

    root = ET.Element("mlt", LC_NUMERIC="C", version="7.0.0", producer="main_bin")

    ET.SubElement(
        root, "profile",
        description=f"{width}x{height} {fps}fps",
        width=str(width), height=str(height),
        frame_rate_num=str(fps), frame_rate_den="1",
        sample_aspect_num="1", sample_aspect_den="1",
        display_aspect_num=str(width), display_aspect_den=str(height),
        progressive="1", colorspace="709",
    )

    # пустая корзина клипов
    main_bin = ET.SubElement(root, "playlist", id="main_bin")
    for name, value in [("shotcut:projectAudioChannels", "2"),
                        ("shotcut:projectFolder", "0"),
                        ("xml_retain", "1")]:
        ET.SubElement(main_bin, "property", name=name).text = value

    # чёрный фон
    black = ET.SubElement(root, "producer", id="black", **{"in": t0, "out": t_end})
    for name, value in [("length", _ts(total_ms)), ("eof", "pause"), ("resource", "0"),
                        ("aspect_ratio", "1"), ("mlt_service", "color"),
                        ("mlt_image_format", "rgba"), ("set.test_audio", "0")]:
        ET.SubElement(black, "property", name=name).text = value
    background = ET.SubElement(root, "playlist", id="background")
    ET.SubElement(background, "entry", producer="black", **{"in": t0, "out": t_end})

    # дорожки: V1 (видео) и A1 (звук)
    video_pl = ET.Element("playlist", id="playlist0")
    ET.SubElement(video_pl, "property", name="shotcut:video").text = "1"
    ET.SubElement(video_pl, "property", name="shotcut:name").text = "V1"

    audio_pl = ET.Element("playlist", id="playlist1")
    ET.SubElement(audio_pl, "property", name="shotcut:audio").text = "1"
    ET.SubElement(audio_pl, "property", name="shotcut:name").text = "A1"

    for i, scene in enumerate(scenes):
        path = os.path.abspath(scene.video_path)
        full_ms = get_duration_ms(path)
        t_in, t_out = _ts(scene.start_ms), _ts(scene.end_ms - 1)  # out включительный

        _add_chain(root, f"v{i}", path, full_ms, video=True)
        _add_chain(root, f"a{i}", path, full_ms, video=False)

        ET.SubElement(video_pl, "entry", producer=f"v{i}", **{"in": t_in, "out": t_out})
        ET.SubElement(audio_pl, "entry", producer=f"a{i}", **{"in": t_in, "out": t_out})

    root.append(video_pl)
    root.append(audio_pl)

    tractor = ET.SubElement(root, "tractor", id="tractor0", **{"in": t0, "out": t_end})
    for name, value in [("shotcut", "1"), ("shotcut:projectAudioChannels", "2"),
                        ("shotcut:projectFolder", "0")]:
        ET.SubElement(tractor, "property", name=name).text = value
    ET.SubElement(tractor, "track", producer="background")
    ET.SubElement(tractor, "track", producer="playlist0")
    ET.SubElement(tractor, "track", producer="playlist1", hide="video")

    def transition(tid, b_track, service, extra=()):
        t = ET.SubElement(tractor, "transition", id=tid)
        for name, value in [("a_track", "0"), ("b_track", b_track), *extra,
                            ("mlt_service", service)]:
            ET.SubElement(t, "property", name=name).text = value

    transition("transition0", "1", "mix", [("always_active", "1"), ("sum", "1")])
    transition("transition1", "1", "qtblend",
               [("compositing", "0"), ("distort", "0"), ("rotate_center", "0"),
                ("threads", "0"), ("disable", "1")])
    transition("transition2", "2", "mix", [("always_active", "1"), ("sum", "1")])

    ET.indent(root)
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)