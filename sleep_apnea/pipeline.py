#!/usr/bin/env python3
"""Complete pipeline for the Sleep Apnea Case Study.

This script runs the full end-to-end workflow:
1. Load data from Synthea output directories into sleep_apnea/data/
2. Print summary statistics for validation
3. Train prediction models
4. Generate comprehensive markdown report

Usage:
    uv run python pipeline.py                    # Full pipeline
    uv run python pipeline.py --skip-data        # Skip data loading (use existing)
    uv run python pipeline.py --skip-models      # Skip model training
    uv run python pipeline.py --stats-only       # Only print summary stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from load_data import process_dataset, resolve_csv_dir, load_sdoh_urban_map
from summary_stats import load_stats, print_summary
from analytics import (
    Dataset as AnalyticsDataset,
    PairwiseStats,
    RegressionStats,
    SummaryStats,
    load_dataset as load_analytics_dataset,
    pairwise_comparison,
    regression_analysis,
)
from models import (
    CrossEvaluation,
    Dataset as ModelDataset,
    ModelResult,
    SplitData,
    load_dataset as load_model_dataset,
    save_models,
    split_dataset,
    train_models,
    _evaluate,
)
from report import (
    FairnessMetrics,
    compute_fairness_metrics,
    write_comprehensive_report,
    _compute_population_stats,
)


def step_load_data(
    baseline_source: Path,
    biased_source: Path,
    reference_source: Path,
    sdoh_path: Path,
    data_dir: Path,
) -> None:
    """Step 1: Load and prepare data from Synthea outputs."""
    print("=" * 60)
    print("Step 1: Loading data from Synthea outputs")
    print("=" * 60)
    print()

    reference_csv_dir = resolve_csv_dir(reference_source)

    print(f"  Baseline source: {baseline_source}")
    print(f"  Biased source: {biased_source}")
    print(f"  Reference headers: {reference_csv_dir}")
    print(f"  SDoH lookup: {sdoh_path}")
    print(f"  Output directory: {data_dir}")
    print()

    process_dataset("baseline", baseline_source, data_dir, reference_csv_dir, sdoh_path)
    process_dataset("biased", biased_source, data_dir, reference_csv_dir, sdoh_path)

    print()
    print("  Data loading complete.")
    print()


def step_summary_stats(data_dir: Path) -> Dict[str, Dict[str, int | float]]:
    """Step 2: Print summary statistics for validation."""
    print("=" * 60)
    print("Step 2: Summary Statistics")
    print("=" * 60)
    print()

    baseline_dir = data_dir / "baseline"
    biased_dir = data_dir / "biased"

    baseline_stats = load_stats(baseline_dir)
    biased_stats = load_stats(biased_dir)

    print_summary(baseline_stats, biased_stats)

    return {"baseline": baseline_stats, "biased": biased_stats}


def step_train_and_report(
    data_dir: Path,
    output_dir: Path,
    model_dir: Path,
    report_path: Path,
    seed: int,
    train_frac: float,
    val_frac: float,
    n_perm: int,
    skip_models: bool,
) -> int:
    """Step 3: Train models and generate comprehensive report."""
    print("=" * 60)
    print("Step 3: Model Training and Report Generation")
    print("=" * 60)
    print()

    baseline_path = data_dir / "baseline"
    biased_path = data_dir / "biased"

    # Load datasets for both analytics and models
    print("[3.1] Loading datasets...")

    analytics_baseline = load_analytics_dataset("baseline", baseline_path)
    analytics_biased = load_analytics_dataset("biased", biased_path)
    analytics_datasets = [analytics_baseline, analytics_biased]

    model_baseline = load_model_dataset("baseline", baseline_path)
    model_biased = load_model_dataset("biased", biased_path)
    model_datasets = [model_baseline, model_biased]

    print(f"  - Baseline: {len(model_baseline.labels):,} patients, prevalence={model_baseline.prevalence:.2%}")
    print(f"  - Biased: {len(model_biased.labels):,} patients, prevalence={model_biased.prevalence:.2%}")
    print()

    # Analytics: Pairwise and regression analysis
    print("[3.2] Running underdiagnosis analysis...")

    pairwise_stats = {
        analytics_baseline.name: pairwise_comparison(analytics_baseline),
        analytics_biased.name: pairwise_comparison(analytics_biased),
    }

    regression_stats = {
        analytics_baseline.name: regression_analysis(analytics_baseline, seed, n_perm),
        analytics_biased.name: regression_analysis(analytics_biased, seed, n_perm),
    }

    population_stats = {
        analytics_baseline.name: _compute_population_stats(analytics_baseline),
        analytics_biased.name: _compute_population_stats(analytics_biased),
    }

    biased_pair = pairwise_stats["biased"]
    print(f"  - Biased rural underdiagnosis rate: {biased_pair.rural_rate:.1%}")
    print(f"  - Biased urban underdiagnosis rate: {biased_pair.urban_rate:.1%}")
    print()

    # Model training
    model_results: List[ModelResult] = []
    cross_results: List[CrossEvaluation] = []
    cross_note = ""
    cross_test_size = 0

    baseline_split = None
    biased_split = None

    if skip_models:
        print("[3.3] Skipping model training (--skip-models)")
        print()
    else:
        print("[3.3] Training prediction models...")

        baseline_split = split_dataset(model_baseline, train_frac, val_frac, seed)

        biased_split = SplitData(
            X_train=model_biased.features.loc[baseline_split.X_train.index],
            X_val=model_biased.features.loc[baseline_split.X_val.index],
            X_test=model_biased.features.loc[baseline_split.X_test.index],
            y_train=model_biased.labels.loc[baseline_split.X_train.index],
            y_val=model_biased.labels.loc[baseline_split.X_val.index],
            y_test=model_biased.labels.loc[baseline_split.X_test.index],
        )

        model_results.extend(
            train_models(model_baseline, seed, train_frac, val_frac, split=baseline_split)
        )
        model_results.extend(
            train_models(model_biased, seed, train_frac, val_frac, split=biased_split)
        )

        print(f"  - Trained {len(model_results)} models")

        # Cross-dataset evaluation
        print("[3.4] Running cross-dataset evaluation...")

        baseline_models = {r.model: r.estimator for r in model_results if r.dataset == "baseline"}
        biased_models = {r.model: r.estimator for r in model_results if r.dataset == "biased"}

        X_cross = baseline_split.X_test
        y_cross = baseline_split.y_test
        cross_test_size = len(y_cross)
        cross_note = (
            f"Both datasets share the same patients (same Synthea seed). A single "
            f"patient-ID split is used so that the biased models are never evaluated "
            f"on patients they trained on. Test N={cross_test_size:,}; "
            f"positives={int(y_cross.sum()):,}."
        )

        for model_name, biased_model in biased_models.items():
            baseline_model = baseline_models.get(model_name)
            if baseline_model is None:
                continue
            baseline_probs = baseline_model.predict_proba(X_cross)[:, 1].tolist()
            biased_probs = biased_model.predict_proba(X_cross)[:, 1].tolist()
            baseline_metrics = _evaluate(y_cross, baseline_probs)
            biased_metrics = _evaluate(y_cross, biased_probs)
            deltas = {key: biased_metrics[key] - baseline_metrics[key] for key in baseline_metrics}
            cross_results.append(
                CrossEvaluation(
                    model=model_name,
                    baseline_metrics=baseline_metrics,
                    biased_metrics=biased_metrics,
                    deltas=deltas,
                )
            )

        print(f"  - Cross-evaluation test size: {cross_test_size}")

        # Save models
        save_models(model_dir, model_results)
        print(f"  - Saved models to {model_dir}")
        print()

    # Fairness analysis
    fairness_results: List[FairnessMetrics] = []

    if model_results and baseline_split is not None and biased_split is not None:
        print("[3.5] Computing fairness metrics...")

        baseline_rural = analytics_baseline.full_population["rural"] if analytics_baseline.full_population is not None else None
        biased_rural = analytics_biased.full_population["rural"] if analytics_biased.full_population is not None else None

        for result in model_results:
            if result.dataset == "baseline" and baseline_rural is not None:
                test_ids = baseline_split.X_test.index
                rural_indicator = baseline_rural.reindex(test_ids)
                valid_mask = rural_indicator.notna()
                if valid_mask.sum() > 0:
                    X_test = baseline_split.X_test[valid_mask]
                    y_test = baseline_split.y_test[valid_mask]
                    rural_test = rural_indicator[valid_mask]
                    fm = compute_fairness_metrics(
                        result.model, result.dataset, result.estimator,
                        X_test, y_test, rural_test,
                    )
                    fairness_results.append(fm)

            elif result.dataset == "biased" and biased_rural is not None:
                test_ids = biased_split.X_test.index
                rural_indicator = biased_rural.reindex(test_ids)
                valid_mask = rural_indicator.notna()
                if valid_mask.sum() > 0:
                    X_test = biased_split.X_test[valid_mask]
                    y_test = biased_split.y_test[valid_mask]
                    rural_test = rural_indicator[valid_mask]
                    fm = compute_fairness_metrics(
                        result.model, result.dataset, result.estimator,
                        X_test, y_test, rural_test,
                    )
                    fairness_results.append(fm)

        print(f"  - Computed fairness metrics for {len(fairness_results)} model-dataset combinations")
        print()

    # Write comprehensive report
    print("[3.6] Writing comprehensive report...")

    write_comprehensive_report(
        report_path,
        model_datasets=model_datasets,
        model_results=model_results,
        train_frac=train_frac,
        val_frac=val_frac,
        cross_results=cross_results,
        cross_note=cross_note,
        cross_test_size=cross_test_size,
        analytics_datasets=analytics_datasets,
        pairwise=pairwise_stats,
        regressions=regression_stats,
        population_stats=population_stats,
        n_perm=n_perm,
        fairness_metrics=fairness_results if fairness_results else None,
    )

    print(f"  - Report written to {report_path}")
    print()

    return 0


def build_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent

    parser = argparse.ArgumentParser(
        description="Complete pipeline for Sleep Apnea Case Study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python pipeline.py                     # Full pipeline
  uv run python pipeline.py --skip-data         # Skip data loading
  uv run python pipeline.py --skip-models       # Skip model training
  uv run python pipeline.py --stats-only        # Only summary stats
""",
    )

    # Data loading arguments
    data_group = parser.add_argument_group("Data Loading")
    data_group.add_argument(
        "--baseline-source",
        default=str(repo_root / "synthea" / "output_baseline"),
        help="Path to baseline Synthea output directory.",
    )
    data_group.add_argument(
        "--biased-source",
        default=str(repo_root / "synthea" / "output_rural_bias"),
        help="Path to biased Synthea output directory.",
    )
    data_group.add_argument(
        "--reference",
        default=str(repo_root / "synthea" / "output"),
        help="Path to reference Synthea output with CSV headers.",
    )
    data_group.add_argument(
        "--sdoh",
        default=str(repo_root / "synthea" / "src" / "main" / "resources" / "geography" / "sdoh.csv"),
        help="Path to SDoH CSV with URBAN column.",
    )
    data_group.add_argument(
        "--data-dir",
        default=str(base_dir / "data"),
        help="Output directory for processed CSV data.",
    )

    # Model training arguments
    model_group = parser.add_argument_group("Model Training")
    model_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    model_group.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Training set fraction.",
    )
    model_group.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation set fraction.",
    )
    model_group.add_argument(
        "--n-perm",
        type=int,
        default=500,
        help="Permutation count for significance testing.",
    )

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        default=str(base_dir / "output"),
        help="Output directory for reports and models.",
    )
    output_group.add_argument(
        "--model-dir",
        default=str(base_dir / "output" / "models"),
        help="Directory to save trained models.",
    )
    output_group.add_argument(
        "--report",
        default=str(base_dir / "output" / "sleep_apnea_report.md"),
        help="Output path for markdown report.",
    )

    # Pipeline control
    control_group = parser.add_argument_group("Pipeline Control")
    control_group.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data loading step (use existing data).",
    )
    control_group.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model training (analytics only).",
    )
    control_group.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print summary statistics, skip everything else.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " Sleep Apnea Case Study: Complete Pipeline ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    report_path = Path(args.report)

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    if not args.skip_data and not args.stats_only:
        step_load_data(
            baseline_source=Path(args.baseline_source),
            biased_source=Path(args.biased_source),
            reference_source=Path(args.reference),
            sdoh_path=Path(args.sdoh),
            data_dir=data_dir,
        )

    # Step 2: Summary stats
    stats = step_summary_stats(data_dir)

    if args.stats_only:
        print("=" * 60)
        print("Done (stats only mode).")
        print("=" * 60)
        return 0

    # Step 3: Train models and generate report
    result = step_train_and_report(
        data_dir=data_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        report_path=report_path,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        n_perm=args.n_perm,
        skip_models=args.skip_models,
    )

    print("=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print()
    print(f"  Report: {report_path}")
    if not args.skip_models:
        print(f"  Models: {model_dir}")
    print()

    return result


if __name__ == "__main__":
    raise SystemExit(main())
