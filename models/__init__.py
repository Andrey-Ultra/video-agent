from .scene import Scene
from .feature import Feature
from PIL import Image

Frame = tuple[int, Image.Image]
Vec = tuple[int, Feature]

from .interfaces import (FeatureExtractor, Clusterer, Exporter, SceneDetector)
