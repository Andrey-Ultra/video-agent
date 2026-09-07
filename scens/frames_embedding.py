from dotenv import load_dotenv
import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

load_dotenv()
FRAME_DIR = '/home/au/PycharmProjects/video/frames'


import matplotlib.pyplot as plt
import matplotlib.patches as patches


def frames_to_matrix(frames: list[Image.Image]):
    model = SentenceTransformer('clip-ViT-B-32')

    embeddings = model.encode(frames, convert_to_tensor=True, show_progress_bar=True)

    emb = embeddings.cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    S = emb @ emb.T

    return S



def find_parts(S, W=8, prominence=0.01):
    from scipy.signal import find_peaks
    N = len(S)
    novelty = np.zeros(N)
    for i in range(W, N - W):
        before = S[i-W:i, i-W:i]
        after = S[i:i+W, i:i+W]
        cross = S[i-W:i, i:i+W]
        novelty[i] = (before.mean() + after.mean()) / 2 - cross.mean()
    peaks, _ = find_peaks(novelty, prominence=prominence, distance=5)
    bounds = [0] + list(peaks) + [N]
    return [(bounds[i], bounds[i+1]) for i in range(len(bounds) - 1)]



def plot_matrix(S, title, save_path, parts=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(S, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Похожесть')
    ax.set_title(title)
    ax.set_xlabel('Кадр')
    ax.set_ylabel('Кадр')
    if parts:
        for start, end in parts:
            rect = patches.Rectangle((start, start), end - start, end - start,
                                      linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
    plt.savefig(save_path, dpi=400)
    plt.show()


def param_sweep(S, save_path='test/sweep.png'):
    Ws = [5, 8, 12, 20]
    proms = [0.01, 0.02, 0.04, 0.06]
    fig, axes = plt.subplots(len(Ws), len(proms), figsize=(4*len(proms), 4*len(Ws)))
    for row, W in enumerate(Ws):
        for col, prom in enumerate(proms):
            parts = find_parts(S, W=W, prominence=prom)
            ax = axes[row][col]
            ax.imshow(S, cmap='viridis')
            for start, end in parts:
                rect = patches.Rectangle((start, start), end-start, end-start,
                                          linewidth=1.5, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            ax.set_title(f'W={W}, prom={prom}\n{len(parts)} сцен', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.show()


def main():
    frame_files = sorted(os.listdir(FRAME_DIR))
    images = [Image.open(os.path.join(FRAME_DIR, f)) for f in frame_files]
    S = frames_to_matrix(images)
    plot_matrix(S, 'Кто на кого похож', 'similarity_matrix.png')
    param_sweep(S)


# def main():
#     frame_files = sorted(os.listdir(FRAME_DIR))
#     images = [Image.open(os.path.join(FRAME_DIR, f)) for f in frame_files]
#     S = frames_to_matrix(images)
#
#     parts = find_parts(S, W=12, prominence=0.06)
#     # plot_matrix(S, f'Границы сцен',
#     #             f'test/similarity_matrix_boxes.png', parts=parts)
#
#     return parts

if __name__ == '__main__':
    main()
