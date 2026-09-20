from export.export_to_shotcut import export_mlt
from frame_embeddings.em_clip import ClipEmbedder
from logging_setup import setup_logging
from models import Scene
from scene_detection.adaptive_em import AdaptiveSplitter
from scene_detection.embedding_detection import split_scene
from scene_detection.cut_detection import split_by_cuts


def main():
    setup_logging()

    embedder = ClipEmbedder(batch_size=16)   # загрузка модели: один раз
    splitter = AdaptiveSplitter(k=4)

    a = Scene.from_video("/Users/au/Documents/git/video-agent/test/fall.mp4")



    shots = split_by_cuts(a)                 # 1. по склейкам


    scenes = []
    for shot in shots:                       # 2. по смыслу, тот же embedder для всех
        scenes += split_scene(shot, embed=embedder, split=splitter)


    print(len(shots))
    print(len(scenes))

    export_mlt(scenes, "/Users/au/Documents/git/video-agent/test/project.mlt")

if __name__ == '__main__':
    main()