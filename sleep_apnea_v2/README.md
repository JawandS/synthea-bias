# Sleep Apnea Case Study v2: Label Masking Bias

This case study demonstrates how barriers to care in rural populations can lead to underdiagnosis of conditions such as sleep apnea, and this structural bias affects machine learning (ML) model performance. This case study generates a baseline population via Synthea then creates a biased dataset by randomly masking sleep apnea diagnoses in rural patients. Finally, it trains and evaluates ML models on both datasets to quantify the impact of underdiagnosis bias.

## Approach

1. **Generate baseline data** - Run Synthea once to create a population with realistic sleep apnea prevalence
2. **Create biased dataset** - Copy baseline data and randomly mask (remove) a percentage of rural patients' sleep apnea diagnoses
3. **Train models** - Train identical models on baseline vs biased data
4. **Measure degradation** - Compare model performance to quantify the impact of underdiagnosis bias

This approach ensures identical patient populations across datasets, isolating the effect of label bias.

## Quick Start

```bash
# Step 1: Generate baseline data from Synthea
uv run python scripts/1_generate_data.py --population 5000 --seed 42

# Step 2: Create biased dataset (TODO)
uv run python scripts/2_create_bias.py --mask-rate 0.5

# Step 3: Train and evaluate models (TODO)
uv run python scripts/3_train_models.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/1_generate_data.py` | Run Synthea, copy CSVs to `data/`, add urban/rural flag |
| `scripts/2_create_bias.py` | Create biased dataset by masking rural diagnoses (TODO) |
| `scripts/3_train_models.py` | Train models and generate comparison report (TODO) |

### Output Files

After running, `data/` contains:

| File | Description |
|------|-------------|
| `patients.csv` | Patient demographics with `urban` column added |
| `conditions.csv` | All patient conditions including sleep apnea diagnoses |
| `observations.csv` | Clinical observations (BMI, smoking status, etc.) |

## Background

### Sleep Apnea in Synthea

The **mofified** Synthea `sleep_apnea.json` module evaluates patients aged 30-60 for sleep apnea risk based on:
- BMI
- Smoking status
- Alcohol use
- Congestive heart failure (CHF)

Patients who enter the pathway receive diagnostic testing (home sleep study or polysomnography) and treatment (CPAP or oral appliance).
