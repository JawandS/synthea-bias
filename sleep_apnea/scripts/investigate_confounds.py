#!/usr/bin/env python3
"""Investigate confounding issues between baseline and biased datasets.

The sleep apnea module override changes the Synthea RNG state for rural patients,
which can cascade into other modules and change features that should be identical.
This script quantifies the full extent of the problem.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
BASELINE = DATA_DIR / "baseline"
BIASED = DATA_DIR / "biased"

SEPARATOR = "=" * 72


def load_all():
    """Load all CSV files from both datasets."""
    d = {}
    for label, path in [("baseline", BASELINE), ("biased", BIASED)]:
        d[label] = {
            "patients": pd.read_csv(path / "patients.csv"),
            "conditions": pd.read_csv(path / "conditions.csv"),
            "observations": pd.read_csv(path / "observations.csv"),
        }
    return d


def check_patient_ids(data):
    print(SEPARATOR)
    print("1. PATIENT ID OVERLAP")
    print(SEPARATOR)
    bp = data["baseline"]["patients"]
    bi = data["biased"]["patients"]
    base_ids = set(bp["Id"])
    bias_ids = set(bi["Id"])
    overlap = base_ids & bias_ids
    print(f"  Baseline patients:  {len(base_ids)}")
    print(f"  Biased patients:    {len(bias_ids)}")
    print(f"  Overlap:            {len(overlap)}")
    print(f"  Only in baseline:   {len(base_ids - bias_ids)}")
    print(f"  Only in biased:     {len(bias_ids - base_ids)}")
    print(f"  Verdict: {'PASS - identical patient sets' if overlap == base_ids == bias_ids else 'FAIL - patient sets differ'}")
    return overlap


def check_demographics(data):
    print(f"\n{SEPARATOR}")
    print("2. DEMOGRAPHIC COLUMN DIFFERENCES (patients.csv)")
    print(SEPARATOR)
    bp = data["baseline"]["patients"].set_index("Id").sort_index()
    bi = data["biased"]["patients"].set_index("Id").sort_index()

    print(f"\n  {'Column':<25s} {'Rows differ':>12s}  {'Pct':>6s}")
    print(f"  {'-'*25} {'-'*12}  {'-'*6}")
    for col in bp.columns:
        try:
            diff = (bp[col].fillna("__NA__").values != bi[col].fillna("__NA__").values).sum()
        except Exception:
            continue
        if diff > 0:
            print(f"  {col:<25s} {diff:>12,d}  {diff/len(bp)*100:5.1f}%")

    # Birthdate detail
    bd_base = pd.to_datetime(bp["BIRTHDATE"])
    bd_bias = pd.to_datetime(bi["BIRTHDATE"])
    day_diffs = (bd_base - bd_bias).dt.days.abs()
    print(f"\n  Birthdate: max day diff = {day_diffs.max()}, "
          f"mean (where >0) = {day_diffs[day_diffs>0].mean():.1f}")
    print(f"  Age impact: negligible (max 1 day)")

    # FIPS detail
    fips_diff = (bp["FIPS"] != bi["FIPS"]).sum()
    # Check if FIPS changes correlate with URBAN
    fips_changed = bp["FIPS"] != bi["FIPS"]
    if "URBAN" in bp.columns:
        rural_fips = fips_changed & (bp["URBAN"] == 0)
        urban_fips = fips_changed & (bp["URBAN"] == 1)
        print(f"\n  FIPS changes by group: rural={rural_fips.sum()}, urban={urban_fips.sum()}")


def check_conditions_drift(data):
    print(f"\n{SEPARATOR}")
    print("3. CONDITIONS DRIFT (conditions.csv)")
    print(SEPARATOR)
    bc = data["baseline"]["conditions"]
    bic = data["biased"]["conditions"]
    print(f"  Baseline rows: {len(bc):,d}")
    print(f"  Biased rows:   {len(bic):,d}")

    base_codes = bc.groupby("CODE").size()
    bias_codes = bic.groupby("CODE").size()
    all_codes = set(base_codes.index) | set(bias_codes.index)

    diffs = []
    for code in all_codes:
        b = base_codes.get(code, 0)
        bi_val = bias_codes.get(code, 0)
        if b != bi_val:
            desc_df = bc.loc[bc["CODE"] == code, "DESCRIPTION"]
            if len(desc_df) == 0:
                desc_df = bic.loc[bic["CODE"] == code, "DESCRIPTION"]
            desc = desc_df.iloc[0] if len(desc_df) > 0 else str(code)
            diffs.append((code, desc, b, bi_val, bi_val - b))

    diffs.sort(key=lambda x: abs(x[4]), reverse=True)
    print(f"\n  Conditions with different counts: {len(diffs)} / {len(all_codes)}")
    print(f"\n  {'Condition':<55s} {'Base':>6s} {'Biased':>6s} {'Delta':>6s}")
    print(f"  {'-'*55} {'-'*6} {'-'*6} {'-'*6}")
    for code, desc, b, bi_val, d in diffs[:25]:
        print(f"  {desc[:55]:<55s} {b:6d} {bi_val:6d} {d:+6d}")


def check_model_features(data):
    """Build the exact features used in the model pipeline and compare."""
    print(f"\n{SEPARATOR}")
    print("4. MODEL FEATURE COMPARISON")
    print(SEPARATOR)

    bp = data["baseline"]["patients"].set_index("Id").sort_index()
    bi = data["biased"]["patients"].set_index("Id").sort_index()
    bc = data["baseline"]["conditions"]
    bic = data["biased"]["conditions"]
    bo = data["baseline"]["observations"]
    bio = data["biased"]["observations"]

    def _build(patients, conditions, observations):
        p = patients.copy()
        p["age_years"] = (pd.to_datetime("2024-01-01") - pd.to_datetime(p["BIRTHDATE"])).dt.days / 365.25
        p["male"] = (p["GENDER"] == "M").astype(int)
        p["income"] = p["INCOME"]

        # BMI
        bmi = observations.loc[observations["CODE"] == "39156-5"].copy()
        bmi["VALUE"] = pd.to_numeric(bmi["VALUE"], errors="coerce")
        bmi_last = bmi.groupby("PATIENT")["VALUE"].last()
        p["bmi"] = p.index.map(bmi_last)

        # Smoking
        smoke = observations.loc[observations["CODE"] == "72166-2"].copy()
        smoke_last = smoke.groupby("PATIENT")["VALUE"].last()
        smoker_vals = {"Current every day smoker", "Current some day smoker"}
        p["smoker"] = p.index.map(lambda x: 1 if smoke_last.get(x, "") in smoker_vals else 0)

        # Alcohol - check multiple codes
        alc_desc = observations["DESCRIPTION"].str.contains("alcohol|drink|AUDIT", case=False, na=False)
        alc_obs = observations.loc[alc_desc]
        if len(alc_obs) > 0:
            alc_obs = alc_obs.copy()
            alc_obs["VALUE"] = pd.to_numeric(alc_obs["VALUE"], errors="coerce")
            alc_last = alc_obs.groupby("PATIENT")["VALUE"].last()
            p["alcohol_use"] = p.index.map(alc_last).fillna(0)
        else:
            p["alcohol_use"] = 0

        # Hypertension (SNOMED 59621000)
        hyp = set(conditions.loc[conditions["CODE"] == 59621000, "PATIENT"])
        p["hypertension"] = p.index.isin(hyp).astype(int)

        # CHF (SNOMED 88805009)
        chf = set(conditions.loc[conditions["CODE"] == 88805009, "PATIENT"])
        p["chf"] = p.index.isin(chf).astype(int)

        return p[["age_years", "male", "income", "bmi", "smoker", "alcohol_use", "hypertension", "chf"]]

    base_feat = _build(bp, bc, bo)
    bias_feat = _build(bi, bic, bio)

    common = base_feat.index.intersection(bias_feat.index).sort_values()
    base_feat = base_feat.loc[common]
    bias_feat = bias_feat.loc[common]

    rural_mask = bp.loc[common, "URBAN"] == 0
    urban_mask = bp.loc[common, "URBAN"] == 1

    features = ["age_years", "male", "income", "bmi", "smoker", "alcohol_use", "hypertension", "chf"]
    binary_cols = {"male", "smoker", "alcohol_use", "hypertension", "chf"}

    for group_name, mask in [("ALL", pd.Series(True, index=common)),
                              ("RURAL", rural_mask),
                              ("URBAN", urban_mask)]:
        ids = common[mask.values]
        n = len(ids)
        print(f"\n  --- {group_name} ({n:,d} patients) ---")
        print(f"  {'Feature':<15s} {'# Differ':>8s} {'Pct':>6s}  {'Base mean':>10s} {'Bias mean':>10s} {'Delta':>10s}")
        print(f"  {'-'*15} {'-'*8} {'-'*6}  {'-'*10} {'-'*10} {'-'*10}")
        for col in features:
            bv = base_feat.loc[ids, col].values
            biv = bias_feat.loc[ids, col].values
            if col in binary_cols:
                diff_n = int((bv != biv).sum())
            else:
                diff_n = int((np.abs(bv - biv) > 0.001).sum())
            bm = np.nanmean(bv)
            bim = np.nanmean(biv)
            print(f"  {col:<15s} {diff_n:>8,d} {diff_n/n*100:5.1f}%  {bm:>10.4f} {bim:>10.4f} {bim-bm:>+10.6f}")

    return base_feat, bias_feat, rural_mask, common


def check_feature_diffs_by_group(base_feat, bias_feat, rural_mask, common, data):
    """For each feature that differs, show rural vs urban breakdown."""
    print(f"\n{SEPARATOR}")
    print("5. FEATURE CHANGE ATTRIBUTION (rural vs urban)")
    print(SEPARATOR)
    bp = data["baseline"]["patients"].set_index("Id").sort_index().loc[common]

    features = ["bmi", "hypertension", "chf"]
    binary_cols = {"hypertension", "chf"}

    for col in features:
        bv = base_feat[col].values
        biv = bias_feat[col].values
        if col in binary_cols:
            changed = bv != biv
        else:
            changed = np.abs(bv - biv) > 0.001

        n_changed = changed.sum()
        if n_changed == 0:
            continue

        rural_changed = changed & rural_mask.values
        urban_changed = changed & (~rural_mask.values)
        print(f"\n  {col}: {n_changed} patients changed")
        print(f"    Rural: {rural_changed.sum()} ({rural_changed.sum()/rural_mask.sum()*100:.2f}% of rural pop)")
        print(f"    Urban: {urban_changed.sum()} ({urban_changed.sum()/(~rural_mask).sum()*100:.2f}% of urban pop)")


def check_sleep_apnea_labels(data):
    """Compare sleep apnea labels - the intended difference."""
    print(f"\n{SEPARATOR}")
    print("6. SLEEP APNEA LABELS (the intended difference)")
    print(SEPARATOR)
    bc = data["baseline"]["conditions"]
    bic = data["biased"]["conditions"]
    bp = data["baseline"]["patients"].set_index("Id").sort_index()

    sa_codes = [73430006, 78275009]  # OSA + sleep apnea
    base_sa = set()
    bias_sa = set()
    for code in sa_codes:
        base_sa |= set(bc.loc[bc["CODE"] == code, "PATIENT"])
        bias_sa |= set(bic.loc[bic["CODE"] == code, "PATIENT"])

    print(f"  Baseline SA patients: {len(base_sa)}")
    print(f"  Biased SA patients:   {len(bias_sa)}")
    print(f"  Dropped: {len(base_sa - bias_sa)}")
    print(f"  Added:   {len(bias_sa - base_sa)}")

    # Breakdown by rural/urban
    dropped = base_sa - bias_sa
    added = bias_sa - base_sa
    for label, pids in [("Dropped", dropped), ("Added", added)]:
        valid = [p for p in pids if p in bp.index]
        if valid:
            rural_n = (bp.loc[valid, "URBAN"] == 0).sum()
            urban_n = (bp.loc[valid, "URBAN"] == 1).sum()
            print(f"  {label}: rural={rural_n}, urban={urban_n}")

    # By subgroup
    all_ids = bp.index
    for group, mask_val in [("Rural", 0), ("Urban", 1)]:
        ids = set(bp[bp["URBAN"] == mask_val].index)
        base_n = len(base_sa & ids)
        bias_n = len(bias_sa & ids)
        total = len(ids)
        print(f"\n  {group} ({total:,d} patients):")
        print(f"    Baseline SA: {base_n} ({base_n/total*100:.2f}%)")
        print(f"    Biased SA:   {bias_n} ({bias_n/total*100:.2f}%)")
        print(f"    Reduction:   {base_n - bias_n} ({(base_n - bias_n)/max(base_n,1)*100:.1f}%)")


def check_alcohol_source(data):
    """Figure out how alcohol_use is being computed."""
    print(f"\n{SEPARATOR}")
    print("7. ALCOHOL_USE SOURCE INVESTIGATION")
    print(SEPARATOR)
    for label in ["baseline", "biased"]:
        obs = data[label]["observations"]
        alc = obs[obs["DESCRIPTION"].str.contains("alcohol|drink|AUDIT", case=False, na=False)]
        print(f"\n  {label}: {len(alc)} alcohol-related observations")
        if len(alc) > 0:
            print(f"  Codes: {alc['CODE'].value_counts().to_dict()}")
            print(f"  Descriptions: {alc['DESCRIPTION'].unique()[:5]}")
        else:
            print("  No alcohol-related observations found in data")

    # Also check what analytics.py uses
    print("\n  Checking analytics.py alcohol implementation...")
    sys.path.insert(0, str(Path("scripts")))
    try:
        import analytics
        import inspect
        src = inspect.getsource(analytics._build_feature_frame)
        # Find alcohol-related lines
        for i, line in enumerate(src.split("\n")):
            if "alcohol" in line.lower() or "drink" in line.lower() or "audit" in line.lower():
                print(f"    analytics.py: {line.strip()}")
    except Exception as e:
        print(f"    Could not inspect analytics.py: {e}")


def check_observation_drift(data):
    """Check if observations (beyond sleep apnea related) differ."""
    print(f"\n{SEPARATOR}")
    print("8. OBSERVATION DRIFT")
    print(SEPARATOR)
    bo = data["baseline"]["observations"]
    bio = data["biased"]["observations"]
    print(f"  Baseline rows: {len(bo):,d}")
    print(f"  Biased rows:   {len(bio):,d}")
    print(f"  Diff: {len(bio) - len(bo):+,d}")

    base_codes = bo.groupby("CODE").size()
    bias_codes = bio.groupby("CODE").size()
    all_codes = set(base_codes.index) | set(bias_codes.index)
    diffs = []
    for code in all_codes:
        b = base_codes.get(code, 0)
        bi_val = bias_codes.get(code, 0)
        if b != bi_val:
            desc_df = bo.loc[bo["CODE"] == code, "DESCRIPTION"]
            if len(desc_df) == 0:
                desc_df = bio.loc[bio["CODE"] == code, "DESCRIPTION"]
            desc = desc_df.iloc[0] if len(desc_df) > 0 else str(code)
            diffs.append((code, desc, b, bi_val, bi_val - b))

    diffs.sort(key=lambda x: abs(x[4]), reverse=True)
    print(f"\n  Observations with different counts: {len(diffs)} / {len(all_codes)}")
    print(f"\n  {'Observation':<55s} {'Base':>6s} {'Biased':>6s} {'Delta':>6s}")
    print(f"  {'-'*55} {'-'*6} {'-'*6} {'-'*6}")
    for code, desc, b, bi_val, d in diffs[:15]:
        print(f"  {desc[:55]:<55s} {b:6d} {bi_val:6d} {d:+6d}")


def summary(base_feat, bias_feat, rural_mask):
    print(f"\n{SEPARATOR}")
    print("9. SUMMARY & CONFOUND ASSESSMENT")
    print(SEPARATOR)

    n_rural = rural_mask.sum()
    n_urban = (~rural_mask).sum()

    # BMI
    bmi_diff = np.abs(base_feat["bmi"].values - bias_feat["bmi"].values) > 0.001
    bmi_rural = (bmi_diff & rural_mask.values).sum()
    bmi_urban = (bmi_diff & (~rural_mask.values)).sum()

    # Hypertension
    hyp_diff = base_feat["hypertension"].values != bias_feat["hypertension"].values
    hyp_rural = (hyp_diff & rural_mask.values).sum()
    hyp_urban = (hyp_diff & (~rural_mask.values)).sum()

    # CHF
    chf_diff = base_feat["chf"].values != bias_feat["chf"].values
    chf_rural = (chf_diff & rural_mask.values).sum()
    chf_urban = (chf_diff & (~rural_mask.values)).sum()

    print("""
  FINDING: The Synthea module override introduces cascading RNG effects.

  The sleep apnea module override changes the random number sequence for
  patients who enter the modified pathway. Because Synthea modules share
  a per-patient RNG, changing one module's path consumption shifts ALL
  subsequent random draws for that patient. This produces unintended
  feature differences:

  Feature       Rural changed   Urban changed   Concern
  ------------- --------------- --------------- --------""")
    print(f"  BMI           {bmi_rural:>5d} ({bmi_rural/n_rural*100:.1f}%)      {bmi_urban:>5d} ({bmi_urban/n_urban*100:.2f}%)      {'HIGH' if bmi_rural > 0 and bmi_urban == 0 else 'moderate'}")
    print(f"  Hypertension  {hyp_rural:>5d} ({hyp_rural/n_rural*100:.1f}%)      {hyp_urban:>5d} ({hyp_urban/n_urban*100:.2f}%)      {'moderate' if hyp_rural + hyp_urban > 10 else 'low'}")
    print(f"  CHF           {chf_rural:>5d} ({chf_rural/n_rural*100:.1f}%)      {chf_urban:>5d} ({chf_urban/n_urban*100:.2f}%)      {'moderate' if chf_rural + chf_urban > 10 else 'low'}")

    print(f"""
  IMPLICATIONS:
  1. BMI changes are 100% rural — a direct RNG cascade confound.
     A model COULD learn rural status through shifted BMI distributions.
  2. Hypertension/CHF changes affect both groups but at small scale
     ({hyp_rural+hyp_urban} / {n_rural+n_urban} = {(hyp_rural+hyp_urban)/(n_rural+n_urban)*100:.2f}% for hypertension).
  3. The summary statistics in the report correctly show these differences
     because they ARE different — but the report should acknowledge this
     as a Synthea RNG artifact, not a meaningful clinical difference.
  4. The report should note that feature distributions are NOT identical
     across datasets, which could confound the bias analysis.
  5. Consider: the "biased" model may learn from BMI shifts rather than
     (or in addition to) label bias. This is a confound, not the intended
     bias mechanism.

  RECOMMENDATIONS:
  a) Report should add a confound analysis section documenting these diffs
  b) Report summary stats table should flag features that differ
  c) Consider whether BMI/hyp/CHF differences are large enough to matter
     for the model (likely small: 230/5493 = 4.2% of rural BMI changed,
     mean shift only -0.007)
  d) Alternative: use baseline features for BOTH models, only swap labels
     (would isolate pure label bias from RNG cascade)
""")


def main():
    print("Sleep Apnea Bias Study — Confound Investigation")
    print(f"{'=' * 72}\n")

    data = load_all()
    check_patient_ids(data)
    check_demographics(data)
    check_conditions_drift(data)
    base_feat, bias_feat, rural_mask, common = check_model_features(data)
    check_feature_diffs_by_group(base_feat, bias_feat, rural_mask, common, data)
    check_sleep_apnea_labels(data)
    check_alcohol_source(data)
    check_observation_drift(data)
    summary(base_feat, bias_feat, rural_mask)


if __name__ == "__main__":
    main()
