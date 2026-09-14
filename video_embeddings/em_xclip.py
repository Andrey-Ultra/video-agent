from transformers import XCLIPProcessor, XCLIPModel
from PIL import Image
import torch
import numpy as np

_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
_processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
_model.eval()

if torch.cuda.is_available():
    _device = "cuda"
elif torch.backends.mps.is_available():
    _device = "mps"
else:
    _device = "cpu"

_model.to(_device)

NUM_FRAMES = 8  # столько ждёт этот чекпоинт X-CLIP


def segment_to_vec(frames: list[Image.Image]):
    idx = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
    sampled = [frames[i] for i in idx]

    inputs = _processor(videos=[sampled], return_tensors="pt").to(_device)
    with torch.no_grad():
        output = _model.get_video_features(**inputs)

    if hasattr(output, "pooler_output"):
        embedding = output.pooler_output
    elif hasattr(output, "video_embeds"):
        embedding = output.video_embeds
    else:
        embedding = output

    return embedding.squeeze().cpu().detach().numpy()


def text_to_vec(text: str):
    inputs = _processor(text=[text], return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        output = _model.get_text_features(**inputs)

    if hasattr(output, "pooler_output"):
        embedding = output.pooler_output
    elif hasattr(output, "text_embeds"):
        embedding = output.text_embeds
    else:
        embedding = output

    return embedding.squeeze().cpu().detach().numpy()