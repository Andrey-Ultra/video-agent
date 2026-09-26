import re

from rapidfuzz import fuzz

_NOISE_RE = re.compile(r"[^a-zа-яё0-9\s]", re.IGNORECASE)


def normalize(text: str) -> str:
    text = text.lower()
    text = _NOISE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a: str, b: str) -> float:
    """Возвращает похожесть двух строк от 0 до 100."""
    return fuzz.ratio(normalize(a), normalize(b))