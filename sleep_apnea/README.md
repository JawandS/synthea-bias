# Sleep Apnea Case Study: Rural Access Bias

This case study demonstrates how healthcare access disparities can be introduced and measured
in synthetic patient data. We compare a baseline Synthea simulation against a biased version
where rural patients experience higher dropout rates in the sleep apnea care pathway.

## Technical overview
- `scripts/load_data.py`: copies the relevant CSV files from Synthea and appends the urban flag to `patients.csv`. Note, this filters `observations.csv` to only include BMI and smoking observations (avoiding large file size).
- `scripts/summary_stats.py`: provides summary statistics on the datasets (should be very similar).
- `scripts/models.py`: trains logistic regression, random forest, and gradient boosted decision tree models to predict sleep apnea diagnosis using the defined features. Implements train/test/validate split and hyperparameter tuning, and writes a markdown report to `sleep_apnea/output`.

## Data Generation

Pick a state with rural counties (e.g., Montana) so the rural branch is exercised.

### Baseline Dataset
```bash
./run_synthea -s 160 -cs 160 -o false -p 20000 \
  --exporter.csv.export=true \
  --exporter.csv.append_mode=true \
  --exporter.baseDirectory=./output_baseline \
  Montana 
```

### Biased Dataset (Rural Access Bias)
```bash
./run_synthea -s 160 -cs 160 -o false -p 20000 \
  --exporter.csv.export=true \
  --exporter.csv.append_mode=true \
  --exporter.baseDirectory=./output_rural_bias \
  --module_override=/home/js/contracts/synthea-bias/config/overrides_rural_sleep_apnea.properties \
  Montana 
```

## Background

### Sleep Apnea Overview

Sleep apnea is a common sleep disorder characterized by repeated interruption of breathing
during sleep. The baseline Synthea module (`sleep_apnea.json`) models the clinical pathway:

1. **Risk Assessment**: Patients aged 30-60 are evaluated based on BMI, smoking, alcohol use, and CHF
2. **Initial Presentation**: Symptoms include loud snoring and excessive daytime sleepiness
3. **Referral**: Primary care refers to sleep specialist
4. **Diagnostic Testing**: Home sleep study or in-lab polysomnography
5. **Treatment**: CPAP therapy or oral appliances with ongoing follow-up

### The Access Bias Problem

In real-world healthcare, rural patients often face barriers to specialty care including:
- Longer travel distances to sleep centers
- Fewer available sleep specialists
- Difficulty scheduling follow-up appointments
- Higher costs of repeated travel

These barriers cause rural patients to "drop out" of the care pathway before receiving
diagnosis and treatment, leading to underdiagnosis and undertreatment of sleep apnea
in rural populations.

## Module Modifications

### Baseline Module Structure

The Synthea `sleep_apnea.json` module includes two states with urban/rural branching:

**Wait Until Overnight Study** (after referral to sleep center):
```
Rural patients (urban == false):
  → Terminal: 0%
  → Overnight Test: 100%
Urban patients:
  → Overnight Test: 100%
```

**Appointment Delay** (after home sleep study):
```
Rural patients (urban == false):
  → Terminal: 0%
  → Follow Up: 100%
Urban patients:
  → Follow Up: 100%
```

In the baseline module, both urban and rural patients have 100% continuation rates.

### Rural Access Bias Override

The override file `config/overrides_rural_sleep_apnea.properties` modifies these transitions
to simulate rural access barriers:

```properties
# Wait Until Overnight Study: 70% of rural patients drop out
sleep_apnea.json::$['states']['Wait Until Overnight Study']['complex_transition'][0]['distributions'][0]['distribution'] = 0.7
sleep_apnea.json::$['states']['Wait Until Overnight Study']['complex_transition'][0]['distributions'][1]['distribution'] = 0.3

# Appointment Delay: 70% of rural patients drop out
sleep_apnea.json::$['states']['Appointment Delay']['complex_transition'][0]['distributions'][0]['distribution'] = 0.7
sleep_apnea.json::$['states']['Appointment Delay']['complex_transition'][0]['distributions'][1]['distribution'] = 0.3
```

This creates a biased dataset where:
- **70% of rural patients** who enter the sleep apnea pathway drop out before diagnosis
- Urban patients continue to receive full care
- The resulting data reflects lower sleep apnea diagnosis rates in rural populations

## Model, Features, Target

### Models
The classifier suite in `scripts/models.py` trains three model families per dataset:
- Logistic regression (with standardized inputs).
- Random forest classifier.
- Gradient boosted decision tree classifier.

Hyperparameters are selected by validation AUC, and final test metrics include AUC, average precision, and Brier score.
Train/val/test default split is 70/15/15.

### Features (no urban/rural)

| Feature | Source | Description |
|---------|--------|-------------|
| `age_years` | `patients.csv` | Patient age at dataset reference date (latest birth/death date in file) |
| `male` | `patients.csv` | Gender indicator (1.0 for male) |
| `income` | `patients.csv` | Annual household income |
| `bmi` | `observations.csv` | Latest recorded BMI (LOINC 39156-5) |
| `smoker` | `observations.csv` | Latest smoking status (LOINC 72166-2) |
| `alcohol_use` | `conditions.csv` | Alcohol use disorder diagnosis (SNOMED 7200002) |
| `hypertension` | `conditions.csv` | Hypertension diagnosis (SNOMED 59621000) |
| `chf` | `conditions.csv` | Congestive heart failure (SNOMED 88805009) |

### Target
Binary label: patient has a sleep apnea diagnosis (SNOMED 73430006 or 78275009) in `conditions.csv`.
