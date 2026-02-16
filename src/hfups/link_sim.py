"""Deterministic link impairment simulator for HFUPS demos."""

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class LinkSimConfig:
    """Configuration for chunking and impairments."""

    drop_rate: float = 0.0
    flip_rate: float = 0.0
    max_chunk: int = 64
    seed: int = 12345


def simulate_link(data: bytes, cfg: LinkSimConfig) -> list[bytes]:
    """Split data into chunks and apply optional drop/bit-flip impairments."""
    if cfg.max_chunk <= 0:
        raise ValueError("max_chunk must be > 0")

    rng = random.Random(cfg.seed)
    chunks: list[bytes] = []
    index = 0

    while index < len(data):
        size = rng.randint(1, cfg.max_chunk)
        chunk = data[index : index + size]
        index += size

        if rng.random() < cfg.drop_rate:
            continue

        if chunk and rng.random() < cfg.flip_rate:
            mutable = bytearray(chunk)
            byte_idx = rng.randrange(len(mutable))
            bit_mask = 1 << rng.randrange(8)
            mutable[byte_idx] ^= bit_mask
            chunk = bytes(mutable)

        chunks.append(chunk)

    return chunks
