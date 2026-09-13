from dataclasses import dataclass

@dataclass
class VideoPoint:
    number: int
    timestamp: float

    def __sub__(self, other: "VideoPoint") -> float:
        result = abs(self.timestamp - other.timestamp)
        return result