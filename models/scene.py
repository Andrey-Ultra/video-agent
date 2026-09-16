from dataclasses import dataclass
from video import get_video_info

@dataclass
class Scene:
    start_frame: int  # Включительно
    end_frame: int  # Невключительно
    fps: float
    video_path: str

    @classmethod
    def from_video(cls, video_path: str) -> "Scene":
        from video.video_parse import get_video_info

        frame_count, fps = get_video_info(video_path)

        return cls(
            start_frame=0,
            end_frame=frame_count,
            fps=fps,
            video_path=video_path,
        )

    @property
    def start_seconds(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_seconds(self) -> float:
        return self.end_frame / self.fps

    @property
    def duration_seconds(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps

    def __lt__(self, other: "Scene") -> bool:
        if not isinstance(other, Scene):
            return NotImplemented
        return self.start_frame < other.start_frame

