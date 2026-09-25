import numpy as np
from models import Feature

class EmVector(Feature):
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def distance(self, other: "ClipEmbedding") -> float:
        cosine_sim = np.dot(self.vector, other.vector) / (
            np.linalg.norm(self.vector) * np.linalg.norm(other.vector)
        )
        return float(1 - cosine_sim)

from .em_clip import ClipEmbedder