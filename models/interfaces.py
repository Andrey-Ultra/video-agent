from typing import Protocol
from models import (Scene, Frame, Vec)


class FeatureExtractor(Protocol):
    def __call__(self, frames: list[Frame]) -> list[Vec]: ...


class Clusterer(Protocol):
    def __call__(self, vectors: list[Vec]) -> list[tuple[int, int]]: ...


class Exporter(Protocol):
    def __call__(self, scenes: list[Scene], path: str) -> None: ...


class SceneDetector(Protocol):
    def __call__(self, scene: Scene) -> list[Scene]: ...