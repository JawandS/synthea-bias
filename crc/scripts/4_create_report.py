#!/usr/bin/env python3
"""Assemble final CRC case-study report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"
INFO_DIR = OUTPUT_DIR / "info"


def load_or_default(path: Path, fallback: str) -> str:
    return path.read_text().strip() if path.exists() else fallback


def main() -> None:
    summary = load_or_default(INFO_DIR / "1_summary_stats.md", "Summary not available.")
    bias = load_or_default(INFO_DIR / "2_bias_effect.md", "Bias report not available.")
    model = load_or_default(INFO_DIR / "3_model.md", "Model report not available.")

    report = f"""# CRC Case Study: Intersectional Bias (Age x Income)

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Study Goal

Evaluate how missing-income-driven label masking distorts a colorectal screening recommendation model
when income is intentionally excluded from training features.

## 1) Data Summary

{summary}

## 2) Bias Construction

{bias}

## 3) Model Comparison

{model}

## 4) Interpretation

- Baseline model is trained on `true_screened_in_last_5y`.
- Biased model is trained on `observed_screened_in_last_5y`.
- Both are evaluated against `true_screened_in_last_5y`.
- Subgroup metrics are computed on the held-out test split from a stratified train/validation/test partition.
- Baseline-vs-biased subgroup recall/F1 deltas reflect information loss induced by biased label capture.
"""

    out_path = OUTPUT_DIR / "report.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
