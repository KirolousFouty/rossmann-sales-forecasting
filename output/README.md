# Output Directory

This directory contains the outputs from running the Rossmann Sales Forecasting pipeline.

## Contents

### Included in Repository
- `plots/` - Visualization images displayed in the main README
- `evaluation_report.txt` - Model performance metrics and comparison

### Generated (Not in Repository)
The following files are generated when you run the pipeline but are excluded from git due to size:

- `*_model.joblib` - Trained model files (~145 MB total)
- `submission.csv` - Kaggle submission file (41,088 predictions)

## Regenerating Outputs

To regenerate all outputs:

```bash
# Train models and generate reports
python src/main.py --generate-submission

# Regenerate plots
python generate_plots.py
```
