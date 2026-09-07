import cv2
from frames_embedding import main as get_parts

VIDEO_PATH = '/home/au/PycharmProjects/video/nepal.mp4'
FRAME_DIR = 'frames'
FPS = 30
OUTPUT_PATH = '/home/au/PycharmProjects/video/nepal-scene.mp4'

COLORS = [
    (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 165, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128),
]  # BGR — так их видит OpenCV, не RGB!


def main():
    parts = get_parts()
    print(f"Найдено сцен: {len(parts)}")
    print(parts)

    scene_of_second = {}
    for scene_idx, (start, end) in enumerate(parts):
        for sec in range(start, end):
            scene_of_second[sec] = scene_idx

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (width, height))

    border_thickness = 15
    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        second = int(frame_idx / FPS)
        scene_idx = scene_of_second.get(second)
        if scene_idx is not None:
            color = COLORS[scene_idx % len(COLORS)]
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, border_thickness)
            cv2.putText(frame, f'Scene {scene_idx + 1}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Сохранено: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()