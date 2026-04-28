"""Configuration objects for chlorophyll-a GDCM-EOF inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class InferenceConfig:
    data_path: Path
    checkpoint_path: Path
    mean_state_path: Path
    output_dir: Path = Path("DINEOF_MODEL_chlor_epoch1_global_output_mask")
    valid_dir: Path = Path("DINEOF_MODEL_chlor_epoch1_global_valid")
    mask_input_dir: Path = Path("DINEOF_MODEL_chlor_epoch1_global_mask_input")
    save_dir: Path = Path("Model_Train_Results")
    device: str = "cuda:1"
    input_channels: int = 7
    year_start: int = 0
    year_end: int = 28
    max_modes: int = 20
    batch_size: int = 1
    load_num: int = 1
    crop_margin: int = 30
    image_height: int = 390
    image_width: int = 900
    num_frequency_bands: int = 16
    outer_epochs: int = 1
    eof_iterations: int = 2

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.image_height, self.image_width
