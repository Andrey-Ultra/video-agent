from models import Scene
from video import extract_frames
from frame_embeddings.em_clip import frames_to_vecs
# todo Сдеалть интеграцию веторизаии прямо в функцию что бы можно было разные эмбединги использовать!
from scene_detection.adaptive_em import adaptive_split
# todo Сдеалть интеграцию сплита так же как и веторизации
import logging

logger = logging.getLogger(__name__)


MAX_BUF_SIZE = 32

def split_scenes(scenes: list[Scene]) -> list[Scene]:
    if not scenes:
        return []

    scenes.sort()

    video_path = scenes[0].video_path
    fps = scenes[0].fps

    index = 0
    buf_frame = []
    buf_vec = []
    ans = []

    logger.info(
        "Начало разбиения: видео=%s, исходных сцен=%d",
        video_path, len(scenes),
    )

    frames = extract_frames(video_path, 1.0)
    num, frame = next(frames, (None, None))

    while index < len(scenes) and num is not None:
        if num < scenes[index].start_frame:
            num, frame = next(frames, (None, None))
        elif scenes[index].start_frame <= num < scenes[index].end_frame:
            buf_frame.append((num, frame))
            if len(buf_frame) >= MAX_BUF_SIZE:
                logger.info("Вычисление эмбеддингов: %d кадров", len(buf_frame))
                buf_vec += frames_to_vecs(buf_frame)
                logger.info("Эмбеддинги готовы: накоплено %d", len(buf_vec))
                buf_frame.clear()

            num, frame = next(frames, (None, None))
        else:
            if buf_frame:
                logger.info("Вычисление эмбеддингов: %d кадров", len(buf_frame))
                buf_vec.extend(frames_to_vecs(buf_frame))
                logger.info("Эмбеддинги готовы: накоплено %d", len(buf_vec))
                buf_frame.clear()

            if buf_vec:
                ans.extend(adaptive_split(buf_vec))
                buf_vec.clear()

            index += 1

    if buf_frame:
        logger.info("Вычисление эмбеддингов: %d кадров", len(buf_frame))
        buf_vec.extend(frames_to_vecs(buf_frame))
        logger.info("Эмбеддинги готовы: накоплено %d", len(buf_vec))
        buf_frame.clear()

    if buf_vec:
        ans.extend(adaptive_split(buf_vec))
        buf_vec.clear()

    scenes_ans = []

    for s, e in ans:
        scenes_ans.append(Scene(start_frame=s, end_frame=e, fps=fps, video_path=video_path))

    logger.info("Разбиение завершено: %d сцен", len(scenes_ans))
    return scenes_ans





def manager(data: list[Scene]) -> list[Scene]:
    scene_dict = dict()
    ans = []

    for scene in data:
        scene_dict[scene.video_path].append(scene)

    for video in scene_dict.keys():

        ans += split_scenes(scene_dict[video])

    return ans
