from peewee import (
    CharField, ForeignKeyField, IntegerField, Model, SqliteDatabase, TextField,
)
import os
from models import Scene


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../video-agent/storage
DB_PATH = os.path.join(BASE_DIR, "..", "video_agent.db")  # .../video-agent/video_agent.db

db = SqliteDatabase(DB_PATH, pragmas={"foreign_keys": 1})


class BaseModel(Model):
    class Meta:
        database = db


class Video(BaseModel):
    path = CharField(unique=True)


class SceneRecord(BaseModel):
    video = ForeignKeyField(Video, backref="scenes", on_delete="CASCADE")
    start_ms = IntegerField()
    end_ms = IntegerField()
    description = TextField(null=True)

    class Meta:
        indexes = ((("video", "start_ms"), True),)

def init_sql_db() -> None:
    db.create_tables([Video, SceneRecord])


def get_or_create_video(path: str) -> int:
    """Возвращает id видео. Если его ещё нет в БД, регистрирует."""
    path = os.path.abspath(path)
    video, created = Video.get_or_create(path=path)
    return video.id



def add_scene(scene: Scene, description: str | None = None) -> int:
    """
    Сохраняет сцену в БД и возвращает её id.
    Если сцена с таким началом в этом видео уже есть, обновляет её.
    """
    video_id = get_or_create_video(scene.video_path)

    record, created = SceneRecord.get_or_create(
        video=video_id,
        start_ms=scene.start_ms,
        defaults={"end_ms": scene.end_ms, "description": description},
    )

    if not created:
        record.end_ms = scene.end_ms
        if description is not None:        # не затираем старое описание пустым
            record.description = description
        record.save()

    return record.id


def get_video_path(video_id: int) -> str:
    """Путь к видео по его id."""
    video = Video.get_or_none(Video.id == video_id)
    if video is None:
        raise ValueError(f"Видео {video_id} не найдено в БД")
    return video.path


def get_scene(scene_id: int) -> tuple[Scene, str | None]:
    """Сцена и её текстовое описание по id."""
    record = (
        SceneRecord
        .select(SceneRecord, Video)     # подтягиваем видео тем же запросом
        .join(Video)
        .where(SceneRecord.id == scene_id)
        .get_or_none()
    )
    if record is None:
        raise ValueError(f"Сцена {scene_id} не найдена в БД")

    scene = Scene(
        start_ms=record.start_ms,
        end_ms=record.end_ms,
        video_path=record.video.path,
    )
    return scene, record.description




# ТЕСТОВЫЙ ФУНКЦИОНАЛ

def _card(record: SceneRecord) -> dict:
    """Единая форма сцены для агента (record должен быть с подтянутым Video)."""
    return {
        "scene_id": record.id,
        "video_id": record.video_id,
        "video": os.path.basename(record.video.path),
        "start_s": round(record.start_ms / 1000, 1),
        "end_s": round(record.end_ms / 1000, 1),
        "duration_s": round((record.end_ms - record.start_ms) / 1000, 1),
        "description": record.description,
    }


def _with_video():
    return SceneRecord.select(SceneRecord, Video).join(Video)


def get_scene_cards(scene_ids: list[int]) -> list[dict]:
    """Карточки нескольких сцен одним запросом, в том же порядке, что и scene_ids."""
    records = _with_video().where(SceneRecord.id.in_(scene_ids))
    by_id = {r.id: r for r in records}
    return [_card(by_id[i]) for i in scene_ids if i in by_id]


def get_scene_card(scene_id: int) -> dict:
    cards = get_scene_cards([scene_id])
    if not cards:
        raise ValueError(f"Сцена {scene_id} не найдена")
    return cards[0]


def count_video_scenes(video_id: int) -> int:
    return SceneRecord.select().where(SceneRecord.video == video_id).count()


def list_video_scenes(video_id: int, offset: int = 0, limit: int = 30) -> list[dict]:
    """Сцены видео по порядку, постранично."""
    records = (
        _with_video()
        .where(SceneRecord.video == video_id)
        .order_by(SceneRecord.start_ms)
        .offset(offset)
        .limit(limit)
    )
    return [_card(r) for r in records]


def find_neighbors(scene_id: int, n: int = 1) -> tuple[list[dict], list[dict]]:
    """(предыдущие, следующие) сцены в том же видео, по n штук."""
    current = SceneRecord.get_or_none(SceneRecord.id == scene_id)
    if current is None:
        raise ValueError(f"Сцена {scene_id} не найдена")

    before = (
        _with_video()
        .where((SceneRecord.video == current.video_id) & (SceneRecord.end_ms <= current.start_ms))
        .order_by(SceneRecord.start_ms.desc())
        .limit(n)
    )
    after = (
        _with_video()
        .where((SceneRecord.video == current.video_id) & (SceneRecord.start_ms >= current.end_ms))
        .order_by(SceneRecord.start_ms)
        .limit(n)
    )
    return [_card(r) for r in reversed(list(before))], [_card(r) for r in after]