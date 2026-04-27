# Parameter-Efficient-Fine-tuning-Foundation-Model-for-Nuclei-Instance-Segmentation
# CAP 5516 Assignment 3 — LoRA Fine-Tuning of MobileSAM for Nuclei Instance Segmentation


## Quick start (Google Colab — recommended)

The entire pipeline is implemented in a single notebook designed to run end-to-end
on Google Colab.

### 1. Open the notebook in Colab

- Download `MobileSAM_LoRA_NuInsSeg.ipynb` from this repo.
- Open Google Colab → **File → Upload notebook** → select the .ipynb.

### 2. Set the runtime to GPU

**Runtime → Change runtime type → Hardware accelerator: GPU**

A T4 is sufficient. L4 or A100 will train faster. CPU mode is not supported.

### 3. Run all cells

**Runtime → Run all**, or run the cells top to bottom (Shift+Enter).

The notebook will:

1. Mount your Google Drive (only used to persist the trained checkpoints and result
   CSVs across sessions; the dataset itself is downloaded to local SSD for speed).
2. Install dependencies (MobileSAM, scikit-image, tifffile, etc.).
3. Download MobileSAM's pretrained weights to Drive.
4. Download the NuInsSeg dataset (~1.6 GB) directly to Colab's local SSD and extract
   it. This takes ~3–5 minutes the first time. The local copy is recreated on each
   new session; the Drive checkpoints persist.
5. Build the dataset index and stratified 5-fold splits (per-organ stratification).
6. Inject LoRA adapters (rank=16, alpha=16) into MobileSAM's TinyViT attention blocks.
7. Train all 5 folds for 100 epochs each (~10 min per fold on T4 GPU; ~50 min total).
   Each fold's best checkpoint is saved to Drive immediately upon completion. If
   Colab disconnects mid-run, simply re-run the training cell — completed folds are
   skipped automatically.
8. Print per-fold and averaged metrics.
9. Generate the trainable-parameter breakdown.
10. Render visual examples (input / ground truth / prediction) and save the figure
    to Drive.

### 4. Find the outputs

After the notebook completes, the following files will be in your Drive at
`MyDrive/NuInsSeg_LoRA/`:

- `checkpoints/mobile_sam.pt` — pretrained MobileSAM weights (downloaded once)
- `checkpoints/fold{0..4}_best.pt` — best LoRA + seg-head weights per fold
- `results/cv_results.csv` — per-fold metrics
- `results/summary.csv` — averaged metrics
- `results/examples_fold0.png` — visual comparison of predicted and ground-truth
  masks

---

## Hyperparameters used

These are set in the training-driver cell:

| Hyperparameter | Value |
|---|---|
| Epochs per fold | 100 |
| Batch size | 2 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| LR scheduler | Cosine annealing |
| LoRA rank (r) | 16 |
| LoRA alpha (α) | 16 |
| Loss | CE (weighted 0.5 / 1.0 / 2.0) + Dice |
| Image input size | 1024 × 1024 |
| Augmentation | Random flips, 90° rotations, light color jitter |
| Cross-validation | 5-fold, stratified per organ |

