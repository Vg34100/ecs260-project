"""Semantic similarity helpers."""
from __future__ import annotations

from typing import List, Optional


def load_codebert_model(device: str = "auto") -> Optional["SentenceTransformer"]:
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except Exception:
        return None

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return SentenceTransformer("microsoft/codebert-base", device=device)


def embed_texts(
    model: "SentenceTransformer",
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 256,
) -> List[List[float]]:
    try:
        return model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            max_length=max_length,
            show_progress_bar=False,
        ).tolist()
    except ValueError:
        return model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ).tolist()
