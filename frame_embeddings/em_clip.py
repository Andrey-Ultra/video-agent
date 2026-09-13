import logging

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from models.embedding import ClipEmbedding

# Загружаем один раз (глобально), а не при каждом вызове —
# иначе будешь заново тянуть веса и инициализировать модель на каждый кадр
_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
_model.eval()

# если есть GPU — сильно ускорит дело
# cuda: NVIDIA (Linux/Windows), mps: Apple Silicon (M1-M4), иначе — CPU
if torch.cuda.is_available():
    _device = "cuda"
elif torch.backends.mps.is_available():
    _device = "mps"
else:
    _device = "cpu"

logging.getLogger(__name__).info("CLIP работает на устройстве: %s", _device)
_model.to(_device)


def frame_to_vec(frame: Image.Image) -> ClipEmbedding:

    inputs = _processor(images=frame, return_tensors="pt").to(_device)
    with torch.no_grad():
        output = _model.get_image_features(**inputs)

    # transformers 5.x возвращает объект, 4.x — сразу тензор
    if hasattr(output, "pooler_output"):
        embedding = output.pooler_output
    elif hasattr(output, "image_embeds"):
        embedding = output.image_embeds
    else:
        embedding = output  # уже тензор — старая версия

    return ClipEmbedding(embedding.squeeze().cpu().detach().numpy())