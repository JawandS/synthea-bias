# Sleep Apnea Case Study v2: Label Masking Bias

Demonstrates how rural underdiagnosis bias affects ML model performance. Generates a baseline population via Synthea, then masks a portion of rural sleep apnea diagnoses to simulate underdiagnosis.

## Quick Start

```bash
# 1. Generate baseline data (~11k requested → ~20k patients)
uv run python scripts/1_generate_data.py -p 11000 -s 42

# 2. Apply bias (30% of rural sleep apnea cases masked)
uv run python scripts/2_gen_bias.py --mask-rate 0.3

# 3. Train and evaluate models (TODO)
uv run python scripts/3_train_models.py
```

## Output Structure

```
output/
├── data/
│   ├── patients.csv      # Demographics + canonical sleep apnea flags
│   ├── conditions.csv    # All conditions (source for diagnoses)
│   └── observations.csv  # BMI, smoking status
└── info/
    ├── summary_stats.md  # Population statistics
    └── bias_effect.md    # Before/after bias analysis
```

## Canonical Sleep Apnea Flags

After running `2_gen_bias.py`, `patients.csv` contains:

| Column | Description |
|--------|-------------|
| `has_sleep_apnea` | **True label** - patient has sleep apnea diagnosis |
| `mask_sleep_apnea` | Whether diagnosis is masked (underdiagnosed) |

**For modeling**: Use `has_sleep_apnea & ~mask_sleep_apnea` as the observed/biased label.

## Scripts

| Script | Purpose |
|--------|---------|
| `1_generate_data.py` | Run Synthea, add `URBAN` flag, output summary stats |
| `2_gen_bias.py` | Add `has_sleep_apnea` and `mask_sleep_apnea` flags |
| `3_train_models.py` | Train/evaluate models on true vs biased labels (TODO) |
