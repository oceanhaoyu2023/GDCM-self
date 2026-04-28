# GDCM-EOF Chlorophyll-a Reconstruction

This repository contains Python code extracted from the original notebook `叶绿素 epoch 1 global mask.ipynb` and organized as reusable modules for GitHub release.

## Structure

```text
src/chlor_gdcm_eof/
  config.py       # Inference configuration dataclass
  data.py         # NumPy dataset loaders and EOF initialization
  encoding.py     # Positional encoding utilities
  eof.py          # EOF/PCA helper used by the dataset
  inference.py    # Global-mask inference workflow
  layers.py       # ConvLSTM, attention, decoder, and residual blocks
  model.py        # GDCMEOFGenerator model
  optimizers.py   # RAdam optimizer from the notebook
scripts/
  run_global_mask_inference.py
```

## Install

```bash
pip install -r requirements.txt
```

For editable local imports:

```bash
pip install -e .
```

## Run Inference

```bash
python scripts/run_global_mask_inference.py   --data-path /data1/表层卫星数据补全/chlor_a_global_0.4   --checkpoint ./Model_Train_Results/model_last_save_revise_epoch_1_back_chlor_global.pth   --mean-state /data1/表层卫星数据补全/mean_state.npy   --device cuda:1
```

The script writes output arrays to:

- `DINEOF_MODEL_chlor_epoch1_global_output_mask/`
- `DINEOF_MODEL_chlor_epoch1_global_valid/`
- `DINEOF_MODEL_chlor_epoch1_global_mask_input/`

## Notes

The notebook imported `func_eofszb` from an external `utils.py` file that was not present in this project folder. A local EOF/PCA implementation is included in `src/chlor_gdcm_eof/eof.py` so the dataset pipeline remains self-contained. If you have the original `utils.py`, compare its EOF behavior before publishing final reproducibility claims.
