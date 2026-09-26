from peewee import (
    CharField, ForeignKeyField, IntegerField, Model, SqliteDatabase, TextField, BigIntegerField, FloatField,
    DateTimeField,
)
import os
db = SqliteDatabase(None)


# =================
#      TABLES
# =================

class BaseModel(Model):
    class Meta:
        database = db


class Video(BaseModel):
    hash = CharField(unique=True)
    path = CharField()                        # последний путь, где видели файл
    duration_ms = IntegerField(null=True)
    width = IntegerField(null=True)
    height = IntegerField(null=True)
    fps = FloatField(null=True)
    created_at = DateTimeField(null=True, index=True)  # когда снято
    lat = FloatField(null=True)
    lon = FloatField(null=True)
    device = CharField(null=True)


class SceneRecord(BaseModel):
    video = ForeignKeyField(Video, backref="scenes", on_delete="CASCADE")
    start_ms = IntegerField()
    end_ms = IntegerField()

    class Meta:
        indexes = ((("video", "start_ms"), True),)


class OcrSpan(BaseModel):
    video = ForeignKeyField(Video, backref="ocr_spans", on_delete="CASCADE")
    start_ms = IntegerField() # Обе границы включительны!
    end_ms = IntegerField()
    text = TextField()


def init_sql_db(db_path: str) -> None:
    db.init(db_path, pragmas={"foreign_keys": 1})
    db.create_tables([Video, SceneRecord, OcrSpan])


# =================
#      ADDERS
# =================

def add_video(video_hash: str, path: str, meta: dict) -> int:
    return Video.create(hash=video_hash, path=path, **meta).id

def add_scene(video_id: int, start_ms: int, end_ms: int) -> int:
    """Добавляет сцену в БД и возвращает её id."""
    return SceneRecord.create(video=video_id, start_ms=start_ms, end_ms=end_ms).id

def add_ocr_span(video_id: int, start_ms: int, end_ms: int, text: str) -> int:
    """Добавляет кусок распознанного текста (границы включительны) и возвращает его id."""
    return OcrSpan.create(video=video_id, start_ms=start_ms, end_ms=end_ms, text=text).id

def add_ocr_spans(video_id: int, spans: list[tuple[int, int, str]]) -> None:
    """Добавляет сразу много кусков [(start_ms, end_ms, text), ...] одной транзакцией."""
    rows = [
        {"video": video_id, "start_ms": start_ms, "end_ms": end_ms, "text": text}
        for start_ms, end_ms, text in spans
    ]
    with db.atomic():
        for i in range(0, len(rows), 100):   # у SQLite есть лимит на число параметров в одном запросе
            OcrSpan.insert_many(rows[i:i + 100]).execute()

# =================
#      GETTERS
# =================

def get_video_id_by_hash(video_hash: str) -> int | None:
    video = Video.get_or_none(Video.hash == video_hash)
    return video.id if video else None

def get_video_id_by_path(path: str) -> int | None:
    video = Video.get_or_none(Video.path == os.path.abspath(path))
    return video.id if video else None

def get_video_path_by_id(video_id: int) -> str | None:
    video = Video.get_or_none(Video.id == video_id)
    return video.path if video else None

def count_scenes(video_id: int) -> int:
    return SceneRecord.select().where(SceneRecord.video == video_id).count()

def list_scenes(video_id: int, offset: int, limit: int) -> list[SceneRecord]:
    return list(
        SceneRecord.select(SceneRecord, Video).join(Video)   # сразу подтягиваем видео, без запроса на каждую сцену
        .where(SceneRecord.video == video_id)
        .order_by(SceneRecord.start_ms)
        .offset(offset)
        .limit(limit)
    )

def get_scene_by_id(scene_id: int) -> SceneRecord | None:
    return SceneRecord.get_or_none(SceneRecord.id == scene_id)

def get_scenes_by_ids(scene_ids: list[int]) -> list[SceneRecord]:
    """Сцены в том же порядке, что и scene_ids; несуществующие id пропускаются."""
    by_id = {
        s.id: s
        for s in SceneRecord.select(SceneRecord, Video).join(Video).where(SceneRecord.id.in_(scene_ids))
    }
    return [by_id[sid] for sid in scene_ids if sid in by_id]

def get_scene_neighbors(scene: SceneRecord, n: int) -> tuple[list[SceneRecord], list[SceneRecord]]:
    previous = list(
        SceneRecord.select()
        .where((SceneRecord.video == scene.video) & (SceneRecord.start_ms < scene.start_ms))
        .order_by(SceneRecord.start_ms.desc())
        .limit(n)
    )[::-1]
    following = list(
        SceneRecord.select()
        .where((SceneRecord.video == scene.video) & (SceneRecord.start_ms > scene.start_ms))
        .order_by(SceneRecord.start_ms)
        .limit(n)
    )
    return previous, following

# =================
#     UPDATER
# =================

def update_video_path(video_id: int, path: str) -> None:
    Video.update(path=path).where(Video.id == video_id).execute()