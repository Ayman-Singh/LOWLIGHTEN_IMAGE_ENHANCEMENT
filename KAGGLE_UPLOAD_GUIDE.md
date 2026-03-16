# ELIEI Kaggle Training — Step-by-Step Upload Guide

## What You Have

| File                            | Location                                           | Purpose                       |
| ------------------------------- | -------------------------------------------------- | ----------------------------- |
| **ELIEI_Kaggle.zip**            | `ELIEI_Implementation/ELIEI_Kaggle.zip` (209 MB)   | All code + train/eval dataset |
| **ELIEI_Kaggle_Training.ipynb** | `ELIEI_Implementation/ELIEI_Kaggle_Training.ipynb` | Kaggle notebook to run        |

## Kaggle Free Tier Info

- **GPU**: T4 or P100 (16 GB VRAM) — 30 hours/week
- **Training time**: ~17 hours for 200K iterations (~0.3s/step on T4)
- **Fits in 1 week** of free quota
- **No disconnects** like Colab — Kaggle runs reliably
- **Output persists** — checkpoints saved even if you close the browser

---

## Step 1: Create a Kaggle Account

1. Go to **https://www.kaggle.com**
2. Click **Register** (top right)
3. Sign up with Google or email
4. **Verify your phone number** — required for GPU access:
   - Go to **https://www.kaggle.com/settings**
   - Under **Phone Verification**, click **Verify**
   - Enter your phone number, receive SMS code, confirm

---

## Step 2: Upload the Dataset

1. Go to **https://www.kaggle.com/datasets/new**
2. Click **New Dataset** (or the **+ Create** button at top)
3. **Dataset title**: `eliei-dataset` (this exact name — the notebook expects it)
4. Click **Upload** and select `ELIEI_Kaggle.zip` (209 MB)
5. Wait for upload to finish
6. **Visibility**: Keep as **Private** (default)
7. Click **Create Dataset**
8. Wait for processing to complete (1-2 minutes)

> **Important**: The dataset name MUST be `eliei-dataset`. If you name it differently,
> update the `INPUT_DIR` variable in Cell 1 of the notebook.

---

## Step 3: Create a Kaggle Notebook

1. Go to **https://www.kaggle.com/code** and click **+ New Notebook**
2. A new notebook opens in the editor

---

## Step 4: Enable GPU Accelerator

1. In the notebook editor, click the **⋮ (three dots)** menu on the right panel
   - Or click on **Session options** right sidebar
2. Under **Accelerator**, select **GPU T4 x2** (or **GPU P100**)
3. Confirm the selection

> **Note**: If you don't see GPU options, verify your phone number (Step 1).

---

## Step 5: Add Your Dataset

1. In the right sidebar, click **+ Add data** (or the "Input" section)
2. Search for `eliei-dataset` (your uploaded dataset)
3. Click the **+** button to add it
4. It will be mounted at `/kaggle/input/eliei-dataset/`

---

## Step 6: Import the Notebook

### Option A — Upload the .ipynb file (Recommended)
1. In the notebook editor, click **File** → **Import Notebook**
2. Select `ELIEI_Kaggle_Training.ipynb` from your computer
3. The cells will load automatically

### Option B — Copy-paste cells manually
1. Open `ELIEI_Kaggle_Training.ipynb` in VS Code or any text editor
2. Copy each cell's code into the Kaggle notebook cells manually

---

## Step 7: Run Training

1. Click **Run All** (▶▶ button at the top) — or run cells one by one:
   - **Cell 1**: Unzips dataset into working directory
   - **Cell 2**: Installs dependencies, verifies GPU
   - **Cell 3**: Checks data structure (should show image counts)
   - **Cell 4**: Smoke test (verifies model builds correctly)
   - **Cell 5**: **MAIN TRAINING** — runs 200K iterations (~17 hours)
   
2. You can **close your browser** while training runs. Kaggle continues in background.
3. Training auto-saves checkpoints every 5000 steps.

---

## Step 8: If Session Disconnects / Quota Runs Out

Kaggle gives 30 hours/week. If training stops midway:

1. Wait for next week's quota to refresh (resets weekly)
2. Open the same notebook
3. **Re-run all cells** — Cell 5 will automatically resume from the latest checkpoint
   - The config has `resume_state: auto` which finds the latest `*_G.pth` checkpoint
   - The `--resume` flag in the training command enables this

### Checkpoint locations (in `/kaggle/working/ELIEI/experiments/ELIEI_IR_RGB/models/`):
- `5000_G.pth`, `10000_G.pth`, ... — periodic saves
- `best_psnr_G.pth` — best validation PSNR

---

## Step 9: Download Results

After training completes:
1. Click the **Output** tab in the notebook
2. All files in `/kaggle/working/` are saved as output
3. Download:
   - `experiments/ELIEI_IR_RGB/models/best_psnr_G.pth` — trained model
   - `results/*.png` — enhanced images (after running test cell)
   - `ELIEI_results.zip` — zipped results (last cell creates this)

---

## Troubleshooting

| Problem                    | Solution                                                                                 |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| "No GPU available"         | Verify phone number at kaggle.com/settings                                               |
| "GPU quota exhausted"      | Wait for weekly reset; training will resume from checkpoint                              |
| "No dataset found"         | Check dataset name is `eliei-dataset`, check it's added to notebook                      |
| "MISSING: data/train/high" | Re-upload zip; dataset may need reprocessing                                             |
| "CUDA out of memory"       | Shouldn't happen with T4 16GB + batch=16 crop=160. If it does, reduce batch to 8 in yaml |

---

## Expected Results

| Metric | Paper  | Expected (200K iters) |
| ------ | ------ | --------------------- |
| PSNR   | 25+ dB | 23-26 dB              |
| SSIM   | 0.85+  | 0.82-0.87             |

Training log will show:
- NLL loss decreasing from ~5-8 to ~2-4
- CAL loss starting at step 500
- Validation PSNR increasing over time
- Best PSNR checkpoint auto-saved
