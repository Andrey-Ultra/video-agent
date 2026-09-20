import logging, sys

def setup_logging(level=logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    for noisy in ("httpx", "urllib3", "PIL", "transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)