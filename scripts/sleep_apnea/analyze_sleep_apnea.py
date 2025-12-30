#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


SLEEP_APNEA_CODES = {"73430006", "78275009"}


def _find_csv_dir(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir():
        if (path / "patients.csv").exists():
            return path
        if (path / "csv" / "patients.csv").exists():
            return path / "csv"
    raise FileNotFoundError(
        f"Unable to locate patients.csv under {path_str}. "
        "Pass the output directory (e.g., output_baseline) or the csv subdir."
    )


def _get_field_value(row, candidates):
    for candidate in candidates:
        for key in row.keys():
            if key.lower() == candidate.lower():
                return row[key]
    return None


def load_patient_ids(patients_path: Path) -> set:
    patient_ids = set()
    with patients_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = _get_field_value(row, ["Id", "ID", "Patient", "PATIENT"])
            if patient_id:
                patient_ids.add(patient_id)
    if not patient_ids:
        raise ValueError(f"No patient IDs found in {patients_path}")
    return patient_ids


def load_sleep_apnea_patients(conditions_path: Path, codes: set) -> set:
    patient_ids = set()
    with conditions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = _get_field_value(row, ["CODE", "Code"])
            if code not in codes:
                continue
            patient_id = _get_field_value(row, ["PATIENT", "Patient", "Id", "ID"])
            if patient_id:
                patient_ids.add(patient_id)
    return patient_ids


def dataset_stats(label: str, base_dir: str, codes: set) -> dict:
    csv_dir = _find_csv_dir(base_dir)
    patients_path = csv_dir / "patients.csv"
    conditions_path = csv_dir / "conditions.csv"

    patient_ids = load_patient_ids(patients_path)
    apnea_patients = load_sleep_apnea_patients(conditions_path, codes)
    apnea_patients = apnea_patients & patient_ids

    total = len(patient_ids)
    apnea_total = len(apnea_patients)
    prevalence = apnea_total / total if total else 0.0

    return {
        "label": label,
        "csv_dir": csv_dir,
        "total_patients": total,
        "sleep_apnea_patients": apnea_total,
        "prevalence": prevalence,
    }


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare sleep apnea prevalence between two Synthea CSV datasets."
    )
    parser.add_argument(
        "--baseline",
        default="output_baseline",
        help="Baseline output directory (or csv subdir). Default: output_baseline",
    )
    parser.add_argument(
        "--biased",
        default="output_rural_bias",
        help="Biased output directory (or csv subdir). Default: output_rural_bias",
    )
    parser.add_argument(
        "--codes",
        default="73430006,78275009",
        help="Comma-separated SNOMED codes for sleep apnea conditions.",
    )
    args = parser.parse_args()

    codes = {code.strip() for code in args.codes.split(",") if code.strip()}

    baseline = dataset_stats("baseline", args.baseline, codes)
    biased = dataset_stats("biased", args.biased, codes)

    print("Sleep apnea prevalence (unique patients with condition):")
    for stats in (baseline, biased):
        print(
            f"- {stats['label']}: {stats['sleep_apnea_patients']}/{stats['total_patients']} "
            f"({_format_pct(stats['prevalence'])}) in {stats['csv_dir']}"
        )

    abs_diff = biased["prevalence"] - baseline["prevalence"]
    rel_diff = None
    if baseline["prevalence"] > 0:
        rel_diff = abs_diff / baseline["prevalence"]

    print("\nDifference (biased - baseline):")
    print(f"- absolute: {_format_pct(abs_diff)}")
    if rel_diff is None:
        print("- relative: n/a (baseline prevalence is 0)")
    else:
        print(f"- relative: {rel_diff * 100:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
