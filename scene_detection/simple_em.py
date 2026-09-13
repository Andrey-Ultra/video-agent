from models.embedding import ClipEmbedding
from models.point import VideoPoint


def simple(vectors: list[tuple[VideoPoint, ClipEmbedding]], p: float ) -> list[tuple[int, int]]:
    if not vectors:
        return []

    scenes = []

    last_embedding = None
    scene_start = vectors[0][0].number  # номер первого кадра

    for point, embedding in vectors:
        if last_embedding is not None:
            if embedding.distance(last_embedding) > p:
                # сцена сменилась — фиксируем конец предыдущей сцены
                scenes.append((scene_start, point.number))
                scene_start = point.number

        last_embedding = embedding

    # не забыть закрыть последнюю сцену
    scenes.append((scene_start, vectors[-1][0].number))

    return scenes