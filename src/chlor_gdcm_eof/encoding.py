"""Positional encoding utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def positional_encoder(
    image_shape: Sequence[int],
    num_frequency_bands: int,
    max_frequencies: Sequence[int] | None = None,
) -> torch.Tensor:
    """Build sinusoidal latitude/longitude-style positional features.

    Returns a tensor with shape ``(*image_shape, 2 * len(image_shape) * num_frequency_bands)``.
    """
    spatial_shape = tuple(image_shape)
    coords = [torch.linspace(-1, 1, steps=s) for s in spatial_shape]
    pos = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=len(spatial_shape))

    if max_frequencies is None:
        max_frequencies = pos.shape[:-1]

    frequencies = [
        torch.linspace(1.0, max_freq / 2.0, num_frequency_bands)
        for max_freq in max_frequencies
    ]
    frequency_grids = [
        pos[..., i : i + 1] * frequencies_i[None, ...]
        for i, frequencies_i in enumerate(frequencies)
    ]
    encodings = [torch.sin(math.pi * grid) for grid in frequency_grids]
    encodings.extend(torch.cos(math.pi * grid) for grid in frequency_grids)
    return torch.cat(encodings, dim=-1)


# Backward-compatible alias for the notebook function name.
PositionalEncoder = positional_encoder
