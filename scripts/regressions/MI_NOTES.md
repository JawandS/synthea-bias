# Overview
Analyzing the MI (Myocardial Infarction) as a potential target for prediction. 

## Relevant files
- `src/main/resources/modules/myocardial_infarction.json`: MI module
- `/home/js/contracts/synthea-bias/output/graphviz/myocardial_infarction.png`: Visualization of the MI module
- `src/main/resources/modules/heart/stemi_pathway.json`: STEMI care pathway
- `src/main/resources/modules/heart/nsteacs_pathway.json`: NSTEMI/UA care pathway
- `src/main/resources/modules/heart/acs_arrival_medications.json`: Arrival meds bundle
- `src/main/resources/modules/heart/acs_discharge_meds.json`: Discharge meds bundle
- `src/main/resources/modules/heart/cardiac_labs.json`: Cardiac labs bundle
- `src/main/java/org/mitre/synthea/modules/CardiovascularDiseaseModule.java`: MI risk (`mi_risk`) calculation

## MI Module Summary
- Monthly hazard: `Chance_of_MI` delays one month and uses patient attribute `mi_risk` to transition to `MI_Onset`.
- Acute event: `MI_Onset` records SNOMED `22298006` (myocardial infarction disorder) and routes 80% to an emergency encounter and 20% to pre-hospital death (STEMI code `401303003`).
- Emergency workup: encounter → cardiac assessment → ECG → arrival meds → labs → troponin → chest x-ray → diagnostic assessment.
- Stratification: diagnostic assessment splits 60% to NSTEACS and 40% to STEMI submodules.
- Post-acute: discharge meds → discharge care plan → end MI → history of MI (`399211009`).
- Recurrence: `Chance_of_Recurrent_MI` loops back to MI onset.

Key attributes set/used:
- `mi_risk` (monthly MI risk, see risk calculation)
- `chance_of_mi_death` (used for mortality after MI)
- `cardiac_surgery_reason` (tracks STEMI/NSTEMI/UA labels into downstream modules)
- `ACS_CABG_referral` (flag for CABG referral)

Key condition codes:
- `22298006` myocardial infarction (disorder)
- `401303003` acute ST segment elevation MI (disorder)
- `401314000` acute non-ST segment elevation MI (disorder) [NSTEACS]
- `4557003` preinfarction syndrome (unstable angina) [NSTEACS]
- `399211009` history of MI (situation)

Key procedure/encounter codes:
- `50849002` emergency room admission
- `710839006` cardiac assessment monitoring device
- `29303009` ECG
- `399208008` chest x-ray
- `165197003` diagnostic assessment
- `433236007` transthoracic echo
- `15220000` lab test
- `89579-7` high-sensitivity troponin I (LOINC observation)

## MI Risk Calculation
- `mi_risk` is computed in `CardiovascularDiseaseModule` as a **monthly** risk derived from a 10-year cardiovascular risk.
- Risk engine uses either ASCVD or Framingham calculators (default is ASCVD in current code).
- Inputs to ASCVD/Framingham include age, sex, smoking, blood pressure, cholesterol, and diabetes status (plus treatment flags depending on calculator).
- `mi_risk` is updated over time and drives the MI module’s monthly transition.

## Related Modules Summary
- `heart/stemi_pathway.json`: STEMI-specific pathway with inpatient admission, cardiology consult, angiography, PCI, and possible CABG via `heart/cabg_sequence`.
- `heart/nsteacs_pathway.json`: NSTEMI/UA pathway; uses troponin to classify NSTEMI vs unstable angina; risk assessment drives invasive vs ischemia-guided strategy, stress testing, angiography, PCI, and possible CABG.
- `heart/acs_arrival_medications.json`: arrival bundle including aspirin, oxygen, nitroglycerin, morphine (all RxNorm-coded meds/procedures).
- `heart/acs_discharge_meds.json`: discharge bundle with aspirin, beta blocker, statin, ACE/ARB, and P2Y12 antiplatelets (calls `medications/beta_blocker`, `medications/statin`, `medications/ace_arb`).
- `heart/cardiac_labs.json`: broad cardiac lab panel (CBC, CMP, lipids, HbA1c, PT/INR, PTT, NT-proBNP, etc.) used in the emergency workup.
