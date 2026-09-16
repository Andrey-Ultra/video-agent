import logging

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from models.embedding import ClipEmbedding

logger = logging.getLogger(__name__)

# Загружаем один раз (глобально), а не при каждом вызове —
# иначе будешь заново тянуть веса и инициализировать модель на каждый кадр
logger.info("Загрузка CLIP")
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

_model.to(_device)
logger.info("CLIP готов, устройство: %s", _device)


def frames_to_vecs(
    frames: list[tuple[int, Image.Image]],
    batch_size: int = 32,
) -> list[tuple[int, ClipEmbedding]]:
    """
    frames: список (frame_number, image)
    возвращает: список (frame_number, embedding) в том же порядке
    """
    results = []

    for i in range(0, len(frames), batch_size):
        chunk = frames[i:i + batch_size]
        chunk_numbers = [num for num, _ in chunk]
        chunk_images = [img for _, img in chunk]

        inputs = _processor(images=chunk_images, return_tensors="pt").to(_device)
        with torch.no_grad():
            output = _model.get_image_features(**inputs)

        if hasattr(output, "pooler_output"):
            batch_embeds = output.pooler_output
        elif hasattr(output, "image_embeds"):
            batch_embeds = output.image_embeds
        else:
            batch_embeds = output

        batch_embeds = batch_embeds.cpu().detach().numpy()

        # zip гарантирует, что i-й номер кадра соответствует i-му эмбеддингу,
        # т.к. порядок в inputs строго соответствует порядку chunk_images
        results.extend(
            (num, ClipEmbedding(vec))
            for num, vec in zip(chunk_numbers, batch_embeds)
        )

    return results
