# Google Colab Setup Guide for Weather Nowcasting Training

## Quick Start Instructions

### Step 1: Upload to Google Drive (First Time Only)
1. Create a folder in your Google Drive called `morocco-weather-nowcasting`
2. Upload the following to this folder:
   - `data/combined/` (all 67 .npz files)
   - `src/` (entire source code folder)

### Step 2: Open Notebook in Google Colab
1. Go to https://colab.research.google.com
2. Click "File" → "Open notebook" → "GitHub"
3. Paste the URL: `https://github.com/YOUR_USERNAME/morocco-weather-nowcasting` (if repo is public)
4. Or manually upload the notebook file

### Step 3: Run Setup Cell (Cell 2)
- This is the first code cell that starts with "GOOGLE COLAB SETUP"
- It will:
  - Mount your Google Drive
  - Install all required packages
  - Clone the repository (if available)
  - Set up data paths
  - Take about 2-3 minutes

### Step 4: Run Training Cells
- Run cells sequentially from cell 3 onwards
- Skip the "⚠️ GOOGLE COLAB SETUP" cells if running locally

### Step 5: Download Results (Cell 3)
- After training completes, run the "Download Results from Colab" cell
- This prepares all results for download

## Alternative: If Git Clone Doesn't Work

Instead of using git clone, do this:

1. **In your local machine**, zip the entire project:
   ```
   project_root/
   ├── src/
   ├── data/
   │   └── combined/
   ├── configs/
   └── notebook/
   ```

2. **Upload to Google Drive** in a folder called `morocco-weather-nowcasting`

3. **Modify the setup cell** to unzip instead of git clone:
   ```python
   # Unzip from Drive instead of git clone
   import zipfile
   drive_zip = '/content/drive/MyDrive/morocco-weather-nowcasting.zip'
   if Path(drive_zip).exists():
       with zipfile.ZipFile(drive_zip, 'r') as zip_ref:
           zip_ref.extractall('/content')
   ```

## Files to Copy Back After Training

After the notebook finishes training, copy these files back to your local machine:

### Essential Files
```
experiments/notebook_training/
├── baseline/
│   ├── baseline_best.pt          ← Trained baseline model
│   ├── forecasts.npz             ← Predictions on test set
│   └── metrics.json              ← Metrics and training history
├── vit/
│   ├── vit_best.pt               ← Trained ViT model
│   ├── forecasts.npz             ← Predictions on test set
│   └── metrics.json              ← Metrics and training history
├── training_curves.png           ← Loss curve visualization
├── metrics_comparison.png        ← Model metrics comparison
├── prediction_comparison.png     ← Sample prediction comparison
└── SUMMARY.txt                   ← Detailed summary report
```

### How to Download from Colab

**Option A: Download from Files Panel**
1. Click Files icon (folder) on left sidebar
2. Navigate to: `/content/colab_results/notebook_training/`
3. Right-click the folder → Download
4. Unzip and place in your local `experiments/` folder

**Option B: Download Zip File**
1. In Files panel, look for `morocco_nowcasting_results.zip`
2. Right-click → Download
3. Unzip locally

**Option C: Copy to Google Drive**
1. The setup already copies to your Google Drive
2. Go to Drive folder: `My Drive/colab_results/notebook_training/`
3. Download from there

## GPU Configuration in Colab

To ensure you're using GPU:
1. Click "Runtime" menu
2. Select "Change runtime type"
3. Set Hardware accelerator to "GPU" (T4 is fine)
4. Click Save

This should reduce training time from ~1 hour (CPU) to ~10 minutes (GPU).

## Expected Training Time on Colab T4 GPU

- Data loading: 1-2 minutes
- Baseline training (5 epochs): 3-4 minutes
- ViT training (3 epochs): 2-3 minutes
- Evaluation & visualization: 2-3 minutes
- **Total: ~10-15 minutes**

## Troubleshooting

### "Module not found" errors
- Make sure all files are properly uploaded to Drive
- Re-run the setup cell

### Out of memory errors
- Reduce `BATCH_SIZE` to 1 in cell 7
- Reduce `BASELINE_EPOCHS` and `VIT_EPOCHS` in cell 7

### Data not found
- Verify `data/combined/` is in the correct Drive folder path
- Check the Drive mount prints correct path

### Training is very slow
- Verify GPU is enabled in Runtime settings
- Check with `torch.cuda.is_available()` in setup cell

## What Each Output File Contains

1. **baseline_best.pt & vit_best.pt**
   - Model weights in PyTorch format
   - Can be loaded later for inference

2. **forecasts.npz**
   - Raw predictions: shape (N_test, T_out, C, H, W)
   - Ground truth targets: same shape
   - Use for custom analysis/visualization

3. **metrics.json**
   - Summary metrics (RMSE, MAE, SSIM, CSI, POD, FAR)
   - Per-channel MAE and RMSE
   - Training/validation loss history

4. **PNG files**
   - Training curves (loss over epochs)
   - Metric comparison bar charts
   - Sample prediction visualization

5. **SUMMARY.txt**
   - Human-readable report with all results
   - Dataset info, model info, key findings

## Next Steps After Getting Results

1. **Local Analysis**: Load the .npz files and analyze predictions
2. **Retraining**: Adjust hyperparameters and retrain
3. **Extended Dataset**: Add more data and retrain for better generalization
4. **Deployment**: Use the trained .pt files for inference

## Questions?

If you have issues:
1. Check the setup cell output for error messages
2. Re-run the setup cell with `!pip install --upgrade` for packages
3. Verify all paths in the error messages
4. Check GPU availability: `!nvidia-smi`

Good luck with training! 🚀
