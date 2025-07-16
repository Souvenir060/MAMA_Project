# MAMA Framework - Quick Start Guide

## 🚀 Reproduce Figure 6 in 3 Steps

### Step 1: Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python verify_installation.py
```

Expected output:
```
🔍 MAMA Framework Installation Verification
==================================================

File structure:
✅ All required files present

Dataset integrity:
✅ Dataset structure correct: 700 train, 150 val, 150 test

Results verification:
✅ MAMA Full MRR verified: 0.8454
✅ MAMA Full NDCG@5: 0.7933

==================================================
🎉 All verification checks passed!
✅ MAMA Framework is ready for use
```

### Step 3: Generate Figure 6
```bash
python src/plot_main_figures.py
```

**Result**: Check `figures/performance_comparison_corrected.pdf` - this is the exact Figure 6 from the paper!

## 📊 Key Results Verification

The generated figure should show:
- **MAMA Full MRR**: 0.845 ± 0.054
- **Performance improvement**: 61.2% over traditional ranking
- **Statistical significance**: p < 0.001 for all comparisons

## 🔧 Optional: Regenerate Results

If you want to regenerate the evaluation results from scratch:

```bash
# This will create a new timestamped results file
python src/run_main_evaluation.py

# Then generate figures from the new results
python src/plot_main_figures.py
```

## 📁 Output Files

After running the scripts, you'll have:
- `figures/performance_comparison_corrected.pdf` - Figure 6 from paper
- `figures/hyperparameter_sensitivity_corrected.pdf` - Hyperparameter analysis
- `figures/statistical_significance_corrected.pdf` - Statistical validation table

## ❓ Troubleshooting

**Q: Import errors?**
A: Run `pip install -r requirements.txt` in your virtual environment

**Q: Figure doesn't match paper?**
A: Ensure you're using the golden results file: `results/final_run_150_test_set_2025-07-04_18-03.json`

**Q: Missing files?**
A: Run `python verify_installation.py` to check what's missing 