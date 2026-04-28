"""Global-mask inference pipeline extracted from the original notebook."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import InferenceConfig
from .data import SSTDatasetInit, SSTDatasetItem
from .encoding import positional_encoder
from .model import GDCMEOFGenerator


def _resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _load_state_dict(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _prepare_position_features(config: InferenceConfig) -> torch.Tensor:
    position_data = positional_encoder(config.image_shape, config.num_frequency_bands)
    position_data = position_data.permute(2, 0, 1)
    position_all = position_data.unsqueeze(0).repeat_interleave(config.input_channels, dim=0)
    position_all = position_all.unsqueeze(0).repeat_interleave(config.load_num, dim=0)
    return position_all.float()


def _prepare_background(config: InferenceConfig, device: torch.device) -> torch.Tensor:
    background = np.log10(np.load(config.mean_state_path)[None, config.crop_margin : -config.crop_margin, :])
    background[np.isnan(background)] = 0
    background_tensor = torch.FloatTensor(background)
    background_tensor = background_tensor[None, None, :].repeat_interleave(config.load_num, dim=0).to(device)
    return background_tensor.repeat_interleave(config.input_channels, dim=1).to(device)


def _output_name(file_name) -> str:
    if isinstance(file_name, (list, tuple)):
        file_name = file_name[0]
    return Path(str(file_name)).name


def run_inference(config: InferenceConfig) -> None:
    """Run the notebook's global-mask inference workflow."""
    device = _resolve_device(config.device)
    for directory in [config.output_dir, config.valid_dir, config.mask_input_dir, config.save_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    model = GDCMEOFGenerator(config.input_channels).to(device)
    model.load_state_dict(_load_state_dict(config.checkpoint_path, device))
    model.eval()
    criterion = nn.MSELoss()
    _ = criterion  # Kept to mirror the notebook setup and ease future training reuse.

    init_dataset = SSTDatasetInit(
        folder_path=config.data_path,
        year=(config.year_start, config.year_end),
        max_modes=config.max_modes,
        crop_margin=config.crop_margin,
    )
    init_loader = DataLoader(init_dataset, batch_size=config.load_num, shuffle=False, drop_last=True)

    position_all = _prepare_position_features(config)
    background = _prepare_background(config, device)
    loss_history: list[float] = []

    with torch.no_grad():
        for _ in range(config.outer_epochs):
            progress_bar = tqdm(init_loader, desc="Inference")
            for input_data_np, input_mask_np, target_data_np, target_mask_np, file_name, input_mask_np_r in progress_bar:
                input_data_np = input_data_np.numpy()
                input_mask_np = input_mask_np.numpy()
                target_data_np = target_data_np.numpy()
                target_mask_np = target_mask_np.numpy()
                input_mask_np_r = input_mask_np_r.numpy()

                for iteration in range(config.eof_iterations):
                    if iteration != 0:
                        continue

                    item_dataset = SSTDatasetItem(
                        input_data=input_data_np,
                        input_mask=input_mask_np,
                        target_data=target_data_np,
                        target_mask=target_mask_np,
                        mean_state_path=config.mean_state_path,
                        max_modes=config.max_modes,
                        crop_margin=config.crop_margin,
                    )
                    item_loader = DataLoader(item_dataset, batch_size=config.load_num, shuffle=True, drop_last=True)

                    for x_input_batch, mask_input_batch, _x_target_batch, _mask_target_batch in item_loader:
                        position_all_batch = position_all[: x_input_batch.shape[0]]
                        position_all_new = mask_input_batch.unsqueeze(2).float() * position_all_batch
                        x_batch_decoder = position_all_batch.to(device)
                        x_batch_decoder = torch.cat((background[: x_input_batch.shape[0]], x_batch_decoder), 2)

                        x_input = torch.cat((x_input_batch, position_all_new), 2).to(device)
                        outputs = model(x_input, x_batch_decoder).cpu().numpy()

                        name = _output_name(file_name)
                        np.save(config.output_dir / name, outputs)
                        np.save(config.valid_dir / name, input_mask_np_r)
                        np.save(config.mask_input_dir / name, input_data_np)

    with (config.save_dir / "loss.json").open("w", encoding="utf-8") as f:
        json.dump(loss_history, f)
