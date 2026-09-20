from typing import Protocol
from PIL import Image
from models.feature import Feature

Frame = tuple[int, Image.Image]
Vec = tuple[int, Feature]


class Embedder(Protocol):
    def __call__(self, frames: list[Frame]) -> list[Vec]: ...


class Splitter(Protocol):
    def __call__(self, vectors: list[Vec]) -> list[tuple[int, int]]: ...