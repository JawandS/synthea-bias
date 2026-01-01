# Sleep Apnea Case Study: Rural Access Bias

This case study demonstrates how healthcare access disparities can be introduced and measured
in synthetic patient data. We compare a baseline Synthea simulation against a biased version
where rural patients experience higher dropout rates in the sleep apnea care pathway.

## Data Generation

Pick a state with rural counties (e.g., Maine) so the rural branch is exercised.

### Baseline Dataset
```bash
./run_synthea -s 160 -cs 160 -o false -p 200000 \
  --exporter.csv.export=true \
  --exporter.csv.append_mode=true \
  --exporter.baseDirectory=./output_baseline \
  Montana 
```

### Biased Dataset (Rural Access Bias)
```bash
./run_synthea -s 160 -cs 160 -o false -p 200000 \
  --exporter.csv.export=true \
  --exporter.csv.append_mode=true \
  --exporter.baseDirectory=./output_rural_bias \
  --module_override=/home/js/contracts/synthea-bias/config/overrides_rural_sleep_apnea.properties \
  Montana 
```

### Run Analysis
```bash
scripts/.venv/bin/python3 scripts/sleep_apnea/main.py \
  --baseline output_baseline \
  --biased output_rural_bias \
  --out scripts/sleep_apnea/sleep_apnea_demand_report.md
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

## Feature Definitions

The demand model uses the following features (intentionally excluding urban/rural and race):

| Feature | Source | Description |
|---------|--------|-------------|
| `age_years` | `patients.csv` | Patient age at dataset end date |
| `male` | `patients.csv` | Gender indicator (1.0 for male) |
| `income` | `patients.csv` | Annual household income |
| `bmi` | `observations.csv` | Latest recorded BMI (LOINC 39156-5) |
| `smoker` | `observations.csv` | Current smoker (LOINC 72166-2) |
| `alcohol_use` | `conditions.csv` | Alcohol use disorder diagnosis (SNOMED 7200002) |
| `hypertension` | `conditions.csv` | Hypertension diagnosis (SNOMED 59621000) |
| `chf` | `conditions.csv` | Congestive heart failure (SNOMED 88805009) |

## Target Definition

Total sleep-related healthcare spend per patient, computed as the sum of:

1. **Encounters**: `TOTAL_CLAIM_COST` from `encounters.csv` where:
   - `REASONCODE` matches a sleep condition code, OR
   - `CODE` matches a sleep-specific encounter code

2. **Procedures**: `BASE_COST` from `procedures.csv` where:
   - `CODE` matches a sleep procedure code, OR
   - `REASONCODE` matches a sleep condition code, OR
   - The procedure is tied to a sleep-related encounter

3. **Medications**: `TOTALCOST` from `medications.csv` where:
   - `REASONCODE` matches a sleep condition code, OR
   - The medication is tied to a sleep-related encounter

4. **Devices/Supplies**: MODE cost from `costs/devices.csv` or `costs/supplies.csv` for:
   - Sleep-related device/supply codes, OR
   - Items tied to sleep-related encounters

### Clinical Codes

**Sleep Conditions (SNOMED-CT)**:
- `39898005`: Sleep disorder
- `73430006`: Sleep apnea (disorder)
- `78275009`: Obstructive sleep apnea syndrome

**Sleep Procedures (SNOMED-CT)**:
- `103750000`: Nocturnal pulse oximetry
- `10563004`: Evaluation of sleep disorder
- `446573003`: CPAP titration
- `60554003`: Polysomnography
- `698560000`: Continuous positive airway pressure monitoring
- `82808001`: Sleep monitoring

**Sleep Encounters (SNOMED-CT)**:
- `185345009`: Encounter for symptom
- `185347001`: Encounter for problem
- `185389009`: Follow-up visit

**Sleep Devices (SNOMED-CT)**:
- `272265001`: General diagnostic medical device
- `701077002`: CPAP unit
- `701100002`: Respiratory monitoring system
- `702172008`: Humidifier
- `706180003`: Respiratory device
- `720253003`: CPAP nasal mask

**Sleep Supplies (SNOMED-CT)**:
- `463659001`: CPAP mask supplies
- `467645007`: CPAP supplies
- `704718009`: CPAP nasal mask
- `706226000`: Continuous positive airway pressure system mask
- `972002`: Air filter

## Modeling Approach

### Demand Prediction (GBDT)

A Gradient Boosted Decision Tree predicts log1p(spend) from clinical features.
This model intentionally excludes urban/rural status to simulate a "geography-blind"
demand model that might be used for resource allocation.

### Bias Quantification (Linear Regression)

A separate linear regression model includes the urban indicator to *measure* the
spending disparity:

```
log1p(spend) ~ age + gender + income + bmi + smoker + alcohol_use + hypertension + chf + urban
```

The urban coefficient represents the spending difference between urban and rural patients
*after controlling for clinical factors*. By comparing this coefficient between baseline
and biased datasets, we quantify the access disparity introduced by rural dropout.

### Statistical Methods

- **Bootstrap resampling** (n=1000): Confidence intervals for all metrics
- **Permutation testing** (n=1000): Hypothesis tests for cross-population differences

## Outputs

- **Findings Report**: `scripts/sleep_apnea/sleep_apnea_demand_report.md`
  - Population statistics and dropout rates
  - Model performance metrics with confidence intervals
  - Geographic bias quantification
  - Statistical significance tests

## Key Insights

1. **The urban coefficient measures disparity, not clinical need**: A positive coefficient
   means urban patients spend more (controlling for demographics), indicating rural
   underutilization of services.

2. **Comparing coefficients reveals bias**: If the biased dataset has a larger urban
   coefficient than baseline, the rural dropout is creating measurable disparity.

3. **A geography-blind model trained on biased data will underpredict rural demand**:
   The GBDT model learns that patients with certain demographics have lower spend,
   but this reflects access barriers, not lower clinical need.

## File Structure

```
scripts/sleep_apnea/
├── README.md                    # This file (study documentation)
├── main.py                      # Main analysis script
├── analyze_sleep_apnea.py       # Analysis functions
├── sleep_apnea_report.py        # Report generation
├── utils.py                     # Data loading utilities
├── sleep_apnea_demand_report.md # Generated findings report
└── artifacts/
    └── base_sleep_apnea.json    # Reference baseline module
```

## References

- Harrison's Principles of Internal Medicine (15th edition, 2001)
- Senaratna et al. (2017). "Prevalence of obstructive sleep apnea in the general population:
  A systematic review." Sleep Medicine Reviews. https://doi.org/10.1016/j.smrv.2016.07.002
