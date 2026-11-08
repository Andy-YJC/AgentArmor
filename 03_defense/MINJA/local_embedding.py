"""Offline CPU embeddings with token windows; no silent long-text truncation."""
from pathlib import Path

import numpy as np


class LocalEmbedding:
    def __init__(self, model_path: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(Path(model_path).resolve()),
                                         device="cpu", local_files_only=True)
        self.window = self.model.max_seq_length - self.model.tokenizer.num_special_tokens_to_add()
        self.calls = 0
        self.windows = 0

    def __call__(self, texts: list[str]) -> np.ndarray:
        result = []
        for text in texts:
            if not text.strip():
                raise ValueError("Embedding input must not be empty")
            tokens = self.model.tokenizer(text, add_special_tokens=False,
                                          return_offsets_mapping=True, verbose=False)
            offsets = tokens["offset_mapping"]
            chunks = []
            start = 0
            while start < len(offsets):
                end = min(start + self.window, len(offsets))
                chunks.append(text[offsets[start][0]:offsets[end - 1][1]])
                if end == len(offsets):
                    break
                start = end - 16
            if not chunks:
                raise ValueError("No tokens in embedding input")
            vectors = self.model.encode(chunks, normalize_embeddings=True,
                                        show_progress_bar=False)
            pooled = vectors.mean(axis=0)
            result.append(pooled / np.linalg.norm(pooled))
            self.windows += len(chunks)
        self.calls += 1
        return np.asarray(result, dtype=np.float32)
