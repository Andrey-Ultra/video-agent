from statistics import median

from models.embedding import ClipEmbedding


def adaptive_split(vectors: list[tuple[int, ClipEmbedding]], k: float = 3.0) -> list[tuple[int, int]]:
    """
    То же самое, что adaptive(), но без VideoPoint — работает напрямую
    с парами (frame_number, embedding), как возвращает frames_to_vecs().
    Порог считается адаптивно по самой выборке: медиана + k * MAD дистанций
    между соседними кадрами.
    """
    if not vectors:
        return []
    if len(vectors) == 1:
        num = vectors[0][0]
        return [(num, num)]

    distances = [vectors[i][1].distance(vectors[i - 1][1]) for i in range(1, len(vectors))]

    med = median(distances)
    mad = median(abs(d - med) for d in distances)
    threshold = med + k * mad * 1.4826  # приводим MAD к масштабу std для нормального распределения

    scenes = []
    scene_start = vectors[0][0]

    for i, distance in enumerate(distances, start=1):
        if distance > threshold:
            num = vectors[i][0]
            scenes.append((scene_start, num))
            scene_start = num

    scenes.append((scene_start, vectors[-1][0]))

    return scenes