# Using this repo on Kaggle

## What to upload as a Kaggle Dataset

REPA weights (~4.9 GB) are not in git. To run on Kaggle:

1. **Create a Kaggle Dataset** (e.g. `repa-pretrained-models`).
2. **Upload** the folder:
   - `REPA/pretrained_models/`
   - Include both `last.pt` and the `flax_ckpt/` directory.
3. In your Kaggle notebook: **Add Dataset** → your dataset.
4. In code, load from:
   - `"/kaggle/input/<your-dataset-name>/last.pt"`
   - `"/kaggle/input/<your-dataset-name>/flax_ckpt/"` (if you use the Flax checkpoint).

You can zip `REPA/pretrained_models` and upload the zip, then unzip in the notebook with one cell if needed.
