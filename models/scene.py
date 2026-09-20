from dataclasses import dataclass, field
from video import get_duration_ms

@dataclass(frozen=True, order=True)
class Scene:
    start_ms: int  # включительно
    end_ms: int    # невключительно
    video_path: str = field(compare=False)

    @classmethod
    def from_video(cls, video_path: str) -> "Scene":
        return cls(start_ms=0, end_ms=get_duration_ms(video_path), video_path=video_path)

    def __post_init__(self):
        if not 0 <= self.start_ms < self.end_ms:
            raise ValueError(f"Некорректная сцена: {self.start_ms}-{self.end_ms}")

    @property
    def start_seconds(self) -> float:
        return self.start_ms / 1000

    @property
    def end_seconds(self) -> float:
        return self.end_ms / 1000

    @property
    def duration_seconds(self) -> float:
        return (self.end_ms - self.start_ms) / 1000
