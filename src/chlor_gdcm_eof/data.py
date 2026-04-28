"""Dataset loaders and EOF initialization for chlorophyll-a reconstruction."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .eof import func_eofszb


class SSTDatasetInit(Dataset):
    """Load yearly input/label NumPy files and prepare masked training samples."""

    def __init__(
        self,
        folder_path: str | os.PathLike,
        year: tuple[int, int] | list[int] = (0, 28),
        max_modes: int = 20,
        crop_margin: int = 30,
        mask_keep_probability: float = 0.8,
        random_seed: int | None = 0,
    ):
        self.folder_path = Path(folder_path)
        self.year = tuple(year)
        self.max_modes = max_modes
        self.crop_margin = crop_margin
        self.mask_keep_probability = mask_keep_probability
        self.random_seed = random_seed
        self.input_path_list = self.load_data_from_folder(self.folder_path, self.year, "input")
        self.mask_path_list = self.load_data_from_folder(self.folder_path, self.year, "label")

    @staticmethod
    def load_data_from_folder(folder_path: Path, year: tuple[int, int], subfolder_name: str) -> list[Path]:
        year_folders = [folder_path / name for name in sorted(os.listdir(folder_path))[year[0] : year[1]]]
        data_paths: list[Path] = []
        for folder in year_folders:
            data_folder = folder / subfolder_name
            data_paths.extend(data_folder / file_name for file_name in os.listdir(data_folder))
        return sorted(data_paths)

    def __len__(self) -> int:
        return len(self.input_path_list)

    def __getitem__(self, idx: int):
        data = np.load(self.input_path_list[idx])
        mask = np.load(self.mask_path_list[idx])

        if self.crop_margin:
            data = data[:, self.crop_margin : -self.crop_margin, :]
            mask = mask[:, self.crop_margin : -self.crop_margin, :]

        data_norm = np.log10(data)
        rng = np.random.default_rng(self.random_seed)
        random_mask = rng.random(mask.shape) < self.mask_keep_probability
        random_mask_r = 1 - random_mask
        input_mask = mask & random_mask
        input_mask_r = mask & random_mask_r
        target_mask = mask

        input_data = np.where(input_mask, data_norm, np.nan)
        input_data_r = np.where(input_mask_r, data_norm, np.nan)
        target_data = np.where(mask, data_norm, 0)

        return input_data, input_mask, target_data, target_mask, str(self.input_path_list[idx]), input_data_r


class SSTDatasetItem(Dataset):
    """Build model-ready tensors after EOF-based low-rank initialization."""

    def __init__(
        self,
        input_data: np.ndarray,
        input_mask: np.ndarray,
        target_data: np.ndarray,
        target_mask: np.ndarray,
        mean_state_path: str | os.PathLike,
        max_modes: int = 20,
        crop_margin: int = 30,
        cumulative_variance_threshold: float = 0.95,
    ):
        self.input_data_np = input_data
        self.input_mask_np = input_mask
        self.target_data_np = target_data
        self.target_mask_np = target_mask
        self.max_modes = max_modes
        self.crop_margin = crop_margin
        self.cumulative_variance_threshold = cumulative_variance_threshold

        coverage = np.load(mean_state_path)
        if self.crop_margin:
            coverage = coverage[self.crop_margin : -self.crop_margin, :]
        self.coverage = np.where(np.isnan(coverage), 0, 1)

    def __len__(self) -> int:
        return self.input_data_np.shape[0]

    def __getitem__(self, idx: int):
        filled_data = self.input_data_np[idx].transpose(1, 2, 0)
        mask_input = self.input_mask_np[idx].transpose(1, 2, 0)
        label_data = self.target_data_np[idx].transpose(1, 2, 0)
        label_mask = self.target_mask_np[idx].transpose(1, 2, 0)

        data_used = filled_data[self.coverage != 0, :]
        mask_used = np.isnan(data_used)
        data = np.copy(data_used)
        data[np.isnan(data)] = 0

        _, _, _, cumulative_variance, _ = func_eofszb(data)
        indices = np.where(cumulative_variance > self.cumulative_variance_threshold)[0]
        maxeof = int(indices[0] + 1) if len(indices) else min(data.shape)
        maxeof = min(maxeof, self.max_modes, min(data.shape))

        if maxeof <= 0:
            output = data.copy()
        else:
            value = np.arange(1, maxeof + 1)
            output = data.copy()
            u, singular_values, vt = np.linalg.svd(output, full_matrices=False)
            eigenvalues = np.diag(singular_values)
            reconstruction = u[:, value - 1] @ eigenvalues[value - 1][:, value - 1] @ vt[value - 1, :]
            output = data * (1 - mask_used) + reconstruction * mask_used

        data_restored = np.full_like(filled_data, 0)
        data_restored[self.coverage != 0, :] = output

        input_x = torch.tensor(data_restored.transpose(2, 0, 1)).float().unsqueeze(1)
        mask_input = torch.tensor(mask_input.transpose(2, 0, 1))
        label_data = torch.tensor(label_data.transpose(2, 0, 1)).float()
        label_mask = torch.tensor(label_mask.transpose(2, 0, 1))
        return input_x, mask_input, label_data, label_mask
