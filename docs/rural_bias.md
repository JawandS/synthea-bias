# Rural Bias Plan: Underutilization via Hypertension Follow-up Dropout

## Goal and Methodology Alignment
- Purpose: create a synthetic dataset where rural patients have systematically lower care utilization due to reduced provider access, matching Scenario 1 in `docs/methodology.md` (Urban / Rural Bias).
- Target population (explicit per methodology): adults in US counties included in `src/main/resources/geography/sdoh.csv`, with utilization captured in claims-like outputs.
- Bias mechanism: rural patients disengage from routine follow-up care after diagnosis, producing fewer outpatient encounters and downstream undercoding.

## Data Hook: Urban/Rural Flag
- `URBAN` is appended to `src/main/resources/geography/sdoh.csv` and becomes `person.attributes['urban']` (column names are lowercased during load).
- Use `urban == true` as the urban subgroup and `urban == false` as the rural subgroup in downstream analysis.

## Specific Module + Override Lever (Concrete, Traceable)
- Module: `src/main/resources/modules/hypertension.json`
- Mechanism: after initial hypertension diagnosis, follow-up care depends on a dropout probability in `Delay_One_Month`.
- We will raise the dropout probability for the rural cohort to model reduced access to routine care.

### Baseline (Current Defaults)
In `Delay_One_Month`:
- Severe hypertension path: Drop Outs = 0.322, Follow-up = 0.678
- Non-severe path: Drop Outs = 0.267, Follow-up = 0.733

### Rural Override (Example Values)
Increase dropout to create underutilization:
- Severe: Drop Outs = 0.50, Follow-up = 0.50
- Non-severe: Drop Outs = 0.45, Follow-up = 0.55

### Override Keys (JSONPath)
Add these to a properties file (see generation steps below):
```
hypertension.json::$['states']['Delay_One_Month']['complex_transition'][1]['distributions'][0]['distribution'] = 0.50
hypertension.json::$['states']['Delay_One_Month']['complex_transition'][1]['distributions'][1]['distribution'] = 0.50
hypertension.json::$['states']['Delay_One_Month']['complex_transition'][2]['distributions'][0]['distribution'] = 0.45
hypertension.json::$['states']['Delay_One_Month']['complex_transition'][2]['distributions'][1]['distribution'] = 0.55
```
Note: both distributions in each pair must sum to 1.0.

## Generation Plan (Two-Run Merge Strategy)
Module overrides are global, so apply them only to the rural cohort via separate runs:

1. **Urban run (baseline)**
   - No override file (or an "urban" file that keeps defaults).
   - Generate N_urban patients.

2. **Rural run (biased)**
   - Use a rural override file with the dropout increases above.
   - Generate N_rural patients.

3. **Filter and merge**
   - Keep only `urban == true` records from the urban run.
   - Keep only `urban == false` records from the rural run.
   - Merge into a single dataset for modeling.

## How to Apply Overrides
- Create a properties file, e.g. `config/overrides_rural.properties`.
- Point Synthea to it via:
  - `module_override=/absolute/path/to/config/overrides_rural.properties` in `src/main/resources/synthea.properties`, or
  - JVM arg: `-Dmodule_override=/absolute/path/to/config/overrides_rural.properties`.
- Optionally generate a full list of override paths via `./gradlew overrides` (writes `output/overrides.properties`) and confirm the JSONPath keys.

## Expected Bias Signals (Stage 1 Diagnostics)
Use the methodology metrics from `docs/methodology.md`:
- **Utilization SMDs**: outpatient visits per year, hypertension follow-up encounters, medication fills.
- **Diagnosis prevalence**: lower hypertension follow-up documentation in rural.
- **Cost concentration**: confirm 5/50 rule still looks plausible after dropout increase.
- **Geographic imbalance**: confirm rural cohort is underrepresented in utilization relative to urban.

Suggested thresholds:
- |SMD| > 0.10 indicates meaningful imbalance (methodology section 3.2).
- KL divergence < 0.10 for non-target distributions; larger values should be explainable by the rural bias design.

## Notes and Extensions
- This plan uses one concrete lever to produce a known rural underutilization pattern. If a stronger effect is needed, apply the same adjustment to later follow-up states in `hypertension.json` (e.g., the `Delay 2_Month` dropout distribution).
- A second lever for preventive care underutilization could be added later using `src/main/resources/modules/wellness_encounters.json`, but start with a single, traceable mechanism for clean attribution.

