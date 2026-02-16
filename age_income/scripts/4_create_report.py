#!/usr/bin/env python3
"""
4_create_report.py - Assemble complete intersectional case study report.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
INFO_DIR = OUTPUT_DIR / "info"
CONFIG_DIR = PROJECT_DIR / "config"


def load_text(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text().strip()
    return fallback


def build_rule_table(rules_path: Path) -> str:
    if not rules_path.exists():
        return "Rules file not found."

    rules = pd.read_csv(rules_path)
    rows = []
    for _, r in rules.iterrows():
        rows.append(
            "| {plan} | ${imin:,.0f} | ${imax:,.0f} | {amin:.0f}-{amax:.0f} | {s:.0f} | {m1:.0%} | {m2:.0%} |".format(
                plan=r["plan_id"],
                imin=r["income_min"],
                imax=r["income_max"],
                amin=r["age_min"],
                amax=r["age_max"],
                s=r["screening_start_age"],
                m1=r["mask_rate_crc"],
                m2=r["mask_rate_early"],
            )
        )

    return "\n".join([
        "| Plan | Income Min | Income Max | Age Range | Screen Start Age | CRC Mask Rate | Early Mask Rate |",
        "|------|------------:|-----------:|----------:|-----------------:|--------------:|----------------:|",
        *rows,
    ])


def main() -> None:
    summary = load_text(INFO_DIR / "1_summary_stats.md", "Summary stats not generated.")
    bias = load_text(INFO_DIR / "2_bias_effect.md", "Bias effect report not generated.")
    model = load_text(INFO_DIR / "3_model.md", "Model report not generated.")
    rules_table = build_rule_table(CONFIG_DIR / "plan_rules.csv")

    report = f"""# Intersectional Case Study: Age + Income Access Bias in Colorectal Cancer

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Scenario

This case study models an intersectional access pattern where age and income jointly determine
screening access generosity. Generous plans screen earlier; restrictive plans screen later.
The underlying disease burden (true CRC) remains unchanged, while observed diagnosis and
observed early-stage detection are masked by plan-specific access barriers.

## Policy Rules

{rules_table}

## Baseline Data Summary

{summary}

## Bias Effect Analysis

{bias}

## Modeling Results

{model}

## Model Inputs Policy

- Model training includes only clinical + demographic features:
  `age, male, bmi, smoker, diabetes, prediabetes, obesity, hypertension, hyperlipidemia, chf`
- Model training excludes direct bias-policy features:
  `income, poverty_ratio, assigned_plan, eligible_for_screening`
- Fairness evaluation is reported by:
  age band, income quintile, and age x income intersections.

## Interpretation Notes

- `has_crc_true` and `has_early_crc_true` are the canonical outcomes derived from stage codes.
- `observed_crc` and `observed_early_crc` represent what appears in biased data after masking.
- Any performance drop from baseline to biased model quantifies information loss induced by
  the age-income access policy.
"""

    out_path = OUTPUT_DIR / "report.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
