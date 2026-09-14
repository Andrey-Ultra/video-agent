from statistics import median

from models.embedding import ClipEmbedding
from models.point import VideoPoint


def adaptive(vectors: list[tuple[VideoPoint, ClipEmbedding]], k: float = 3.0) -> list[tuple[int, int]]:
    """
    Как simple(), но порог не задаётся константой, а считается по самой
    выборке: сцена меняется там, где дистанция между соседними кадрами
    аномальна относительно типичного разброса дистанций именно этого видео.

    Берём медиану и MAD (median absolute deviation) вместо среднего и std —
    они устойчивы к выбросам, то есть сами резкие склейки (которые мы и
    ищем) не утягивают порог вверх, как это было бы со средним/std.
    """
    if not vectors:
        return []
    if len(vectors) == 1:
        return [(vectors[0][0].number, vectors[0][0].number)]

    distances = [vectors[i][1].distance(vectors[i - 1][1]) for i in range(1, len(vectors))]

    med = median(distances)
    mad = median(abs(d - med) for d in distances)
    threshold = med + k * mad * 1.4826  # 1.4826 приводит MAD к масштабу std для нормального распределения

    scenes = []
    scene_start = vectors[0][0].number

    for i, distance in enumerate(distances, start=1):
        if distance > threshold:
            point = vectors[i][0]
            # сцена сменилась — фиксируем конец предыдущей сцены
            scenes.append((scene_start, point.number))
            scene_start = point.number

    # не забыть закрыть последнюю сцену
    scenes.append((scene_start, vectors[-1][0].number))

    return scenes
