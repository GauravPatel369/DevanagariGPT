"""Batching over the Phase-1 uint16 token memmaps.

Phase 1 wrote each split as one flat ``uint16`` array with documents separated
by the EOS id, which is exactly the shape a causal LM wants: a training example
is a contiguous slice, and no per-document padding or bucketing is needed.

The whole file is *never* read into RAM -- ``np.memmap`` leaves it on disk and
the OS pages in only the windows actually touched.  Hindi's ``train.bin`` is
901 MB, which would otherwise not fit alongside the model on a 6 GB card.

**The shift is the load-bearing detail.**  ``x`` is ``tokens[i : i+T]`` and
``y`` is ``tokens[i+1 : i+T+1]``, so the target at position t is the token at
t+1.  Getting this wrong -- passing the same slice as both -- does not raise;
it asks the model to predict the token it can already see, which with tied
embeddings it does well even at initialisation (see
``tests/test_model.py::test_predicting_the_current_token_is_trivially_easy``).
The result is a loss curve that looks excellent and a model that has learnt
nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

DTYPE = np.uint16


class TokenDataset:
    """Random-access windows over one split's ``.bin`` file.

    Args:
        path: ``<lang>/data/tokens/<split>.bin`` from Phase 1.
        context: window length T; batches are ``(B, T)``.
    """

    def __init__(self, path: str | Path, context: int):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found -- Phase-1 encode stage must run first")
        self.context = context
        self.tokens = np.memmap(self.path, dtype=DTYPE, mode="r")
        # Need one extra token for the shifted target of the final window.
        self.n_windows = len(self.tokens) - context - 1
        if self.n_windows <= 0:
            raise ValueError(f"{self.path} holds {len(self.tokens)} tokens, too few for context {context}")

    def __len__(self) -> int:
        return self.n_windows

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def batch(self, batch_size: int, generator: np.random.Generator, device=None):
        """Sample a random batch.

        Offsets are drawn uniformly rather than walked in order: sequential
        windows overlap heavily and would make consecutive gradient steps highly
        correlated. Uniform sampling over ~450M positions makes a repeat within
        one epoch vanishingly unlikely.
        """
        offsets = generator.integers(0, self.n_windows, size=batch_size)
        return self._gather(offsets, device)

    def deterministic_batches(self, batch_size: int, n_batches: int, seed: int = 1337, device=None):
        """Fixed batches for evaluation.

        Validation loss is only comparable across checkpoints if every
        evaluation sees the same windows, so the offsets come from a seeded
        generator rather than the training stream.
        """
        gen = np.random.default_rng(seed)
        for _ in range(n_batches):
            yield self._gather(gen.integers(0, self.n_windows, size=batch_size), device)

    def sequential_batches(self, batch_size: int, device=None, limit_tokens: int | None = None):
        """Non-overlapping windows in file order, for full-split evaluation.

        Perplexity and bits-per-byte quoted in the report must cover the split
        once, without double-counting: random sampling would weight some tokens
        more than others.
        """
        stride = self.context
        starts = np.arange(0, self.n_windows, stride)
        if limit_tokens is not None:
            starts = starts[: max(1, limit_tokens // stride)]
        for i in range(0, len(starts), batch_size):
            chunk = starts[i : i + batch_size]
            if len(chunk) == 0:
                continue
            yield self._gather(chunk, device)

    def _gather(self, offsets, device):
        T = self.context
        # int64 for the embedding lookup; the file is uint16 to halve its size.
        x = np.stack([self.tokens[o : o + T] for o in offsets]).astype(np.int64)
        y = np.stack([self.tokens[o + 1 : o + 1 + T] for o in offsets]).astype(np.int64)
        x_t, y_t = torch.from_numpy(x), torch.from_numpy(y)
        if device is not None and str(device).startswith("cuda"):
            # pin_memory + non_blocking overlaps the host->device copy with compute.
            x_t = x_t.pin_memory().to(device, non_blocking=True)
            y_t = y_t.pin_memory().to(device, non_blocking=True)
        elif device is not None:
            x_t, y_t = x_t.to(device), y_t.to(device)
        return x_t, y_t


def _walk_kaggle_input(marker_rel: str, maxdepth: int = 6) -> Path | None:
    """Find a directory under /kaggle/input containing ``marker_rel``.

    Kaggle's mount layout is not stable -- it may be ``/kaggle/input/<slug>/`` or
    ``/kaggle/input/datasets/<user>/<slug>/`` -- so the location is discovered by
    walking for a marker file rather than assembled from a hardcoded path that
    silently resolves to an empty mount.
    """
    import os

    base = Path("/kaggle/input")
    if not base.exists():
        return None
    for root_dir, dirs, _files in os.walk(base):
        if root_dir.count(os.sep) - 2 > maxdepth:
            dirs[:] = []
            continue
        if (Path(root_dir) / marker_rel).exists():
            return Path(root_dir)
    return None


def resolve_data_dir(lang: str, root: str | Path = ".") -> Path:
    """Locate the token directory, locally or on Kaggle."""
    root = Path(root)
    local = root / lang / "data" / "tokens"
    if (local / "train.bin").exists():
        return local
    found = _walk_kaggle_input(f"{lang}/train.bin")
    if found is not None:
        return found / lang
    raise FileNotFoundError(
        f"no {lang}/train.bin found under {local} or /kaggle/input"
    )


def resolve_tokenizer(lang: str, root: str | Path = ".") -> Path:
    """Locate the SentencePiece model for one language.

    Locally it sits in ``<lang>/tokenizer/``; on Kaggle it was uploaded
    alongside the token binaries, so both are checked.
    """
    root = Path(root)
    name = f"{'hi' if lang == 'hindi' else 'ne'}_unigram_10000.model"
    local = root / lang / "tokenizer" / name
    if local.exists():
        return local
    found = _walk_kaggle_input(f"{lang}/{name}")
    if found is not None:
        return found / lang / name
    raise FileNotFoundError(f"no {name} found under {local} or /kaggle/input")
