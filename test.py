from export.export_to_shotcut import export_mlt
from frame_embeddings.em_clip import ClipEmbedder
from logging_setup import setup_logging
from models import Scene
from scene_detection.adaptive_em import AdaptiveSplitter
from scene_detection.embedding_detection import split_scene
from scene_detection.cut_detection import split_by_cuts

from describe.fast_describer import describe_scene_fast

from storage import *
from video_embeddings.em_xclip import scene_to_vec
from storage.vector_store import add_scene_vector

def main():
    setup_logging()
    init_sql_db()

    mas = ["/Users/au/Documents/git/video-agent/test/scot1.mp4",
           "/Users/au/Documents/git/video-agent/test/scot2.mp4",
           "/Users/au/Documents/git/video-agent/test/scot3.mp4"]

    for p in mas:

        a = Scene.from_video(p)

        scenes = split_by_cuts(a)


        for scene in scenes:
            scene_id = add_scene(scene, describe_scene_fast(scene))
            vec = scene_to_vec(scene)
            add_scene_vector(scene_id, vec)

            print(scene_id)


if __name__ == '__main__':
    main()