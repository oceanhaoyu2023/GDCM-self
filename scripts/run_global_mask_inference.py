"""Command-line entry point for global-mask chlorophyll-a inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from chlor_gdcm_eof.config import InferenceConfig
from chlor_gdcm_eof.inference import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GDCM-EOF global-mask inference.")
    parser.add_argument("--data-path", type=Path, required=True, help="Dataset root containing yearly input/label folders.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path.")
    parser.add_argument("--mean-state", type=Path, required=True, help="Mean-state .npy path used to build the coverage mask.")
    parser.add_argument("--output-dir", type=Path, default=Path("DINEOF_MODEL_chlor_epoch1_global_output_mask"))
    parser.add_argument("--valid-dir", type=Path, default=Path("DINEOF_MODEL_chlor_epoch1_global_valid"))
    parser.add_argument("--mask-input-dir", type=Path, default=Path("DINEOF_MODEL_chlor_epoch1_global_mask_input"))
    parser.add_argument("--save-dir", type=Path, default=Path("Model_Train_Results"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--year-start", type=int, default=0)
    parser.add_argument("--year-end", type=int, default=28)
    parser.add_argument("--max-modes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-num", type=int, default=1)
    parser.add_argument("--crop-margin", type=int, default=30)
    parser.add_argument("--image-height", type=int, default=390)
    parser.add_argument("--image-width", type=int, default=900)
    parser.add_argument("--num-frequency-bands", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InferenceConfig(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        mean_state_path=args.mean_state,
        output_dir=args.output_dir,
        valid_dir=args.valid_dir,
        mask_input_dir=args.mask_input_dir,
        save_dir=args.save_dir,
        device=args.device,
        year_start=args.year_start,
        year_end=args.year_end,
        max_modes=args.max_modes,
        batch_size=args.batch_size,
        load_num=args.load_num,
        crop_margin=args.crop_margin,
        image_height=args.image_height,
        image_width=args.image_width,
        num_frequency_bands=args.num_frequency_bands,
    )
    run_inference(config)


if __name__ == "__main__":
    main()
