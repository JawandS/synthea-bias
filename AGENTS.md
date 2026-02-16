# Repository Guidelines

## Project Structure & Module Organization
This repository combines a Java simulation engine with Python bias-analysis pipelines.

- `synthea/`: core Synthea engine (Gradle, Java source in `src/main/java`, tests in `src/test/java`, configs in `config/`).
- `diabetes_v2/` and `sleep_apnea_v2/`: end-to-end case studies with scripts in `scripts/`, generated artifacts in `output/`, and project metadata in `pyproject.toml`.
- `flag_urban/`: utilities and reference data for urban/rural labeling.
- `age_income/`: lightweight Python project scaffold.
- `setup.sh`: local machine bootstrap for Java/Gradle setup.

Keep generated outputs under each project’s `output/` folder; avoid mixing outputs across case studies.

## Build, Test, and Development Commands
- `cd synthea && ./gradlew clean build test`: compile Java code and run the full test suite.
- `cd synthea && ./run_synthea -h`: view simulator CLI options.
- `cd diabetes_v2 && uv run python scripts/1_generate_data.py -p 20000 -s 160`: generate baseline diabetes dataset.
- `cd diabetes_v2 && uv run python scripts/2_gen_bias.py && uv run python scripts/3_train_models.py && uv run python scripts/4_create_report.py`: run diabetes bias pipeline.
- `cd sleep_apnea_v2 && uv run python scripts/1_generate_data.py -p 11000 -s 42`: generate sleep apnea baseline, then run scripts `2` through `4`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints where practical, `snake_case` for functions/variables, descriptive script names (`1_generate_data.py`, `2_gen_bias.py`).
- Java: follow Google-style conventions enforced by Checkstyle (`synthea/config/checkstyle/checkstyle.xml`), with `PascalCase` classes and `camelCase` methods.
- Prefer small, single-purpose functions and explicit constants for clinical codes.

## Testing Guidelines
- Java tests use JUnit 4 via Gradle; run from `synthea/` with `./gradlew test`.
- Coverage reports are produced via JaCoCo during `check`/`build`.
- Python case studies currently rely on pipeline-level validation; rerun scripts and confirm updated `output/info/*.md` and `output/report.md` are consistent.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects (for example: `refine case study`, `fix analysis`). Keep subjects concise and scoped.

For PRs:
- Describe the dataset or module impacted.
- List exact commands used to validate changes.
- Include before/after metrics or report snippets when model behavior changes.
- Link related issues and call out any regenerated outputs.
