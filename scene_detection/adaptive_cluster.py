from statistics import median

from models import Vec


class AdaptiveClusterer:
    """
    Реализация Splitter: режет сцены по скачкам расстояния между соседними кадрами.
    Порог считается адаптивно по самой выборке: медиана + k * MAD.
    Чем больше k, тем реже режет (меньше чувствительность).
    """

    def __init__(self, k: float = 3.0):
        self.k = k

    def __call__(self, vectors: list[Vec]) -> list[tuple[int, int]]:
        if not vectors:
            return []
        if len(vectors) == 1:
            t = vectors[0][0]
            return [(t, t)]

        distances = [vectors[i][1].distance(vectors[i - 1][1]) for i in range(1, len(vectors))]

        med = median(distances)
        mad = median(abs(d - med) for d in distances)
        threshold = med + self.k * mad * 1.4826  # MAD приведён к масштабу std

        scenes = []
        scene_start = vectors[0][0]

        for i, distance in enumerate(distances, start=1):
            if distance > threshold:
                t = vectors[i][0]
                scenes.append((scene_start, t))
                scene_start = t

        scenes.append((scene_start, vectors[-1][0]))

        return scenes