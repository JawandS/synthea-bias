# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository combines **Synthea** (a synthetic patient population simulator in Java) with **Python case studies** demonstrating various forms of bias in healthcare datasets and ML models. The project generates realistic patient data, then applies bias masking to study how underdiagnosis and documentation bias affect model performance.

**Requirements**: Java JDK 11 or 17 (LTS), Python 3.11+, uv (for Python dependency management)

## Repository Structure

```
synthea-bias/
├── synthea/                # Core Synthea simulator (Java/Gradle)
├── diabetes_v2/            # Documentation bias case study (Python)
├── sleep_apnea_v2/         # Rural underdiagnosis bias case study (Python)
├── flag_urban/             # Urban/rural classification helper
└── age_income/             # Age/income bias exploration (in development)
```

## Build and Run Commands

### Synthea (Java) - run from `synthea/` directory

```bash
./gradlew build              # Build the project
./gradlew test               # Run all tests
./gradlew test --tests "org.mitre.synthea.engine.ModuleTest"  # Specific test class
./gradlew check              # Tests + checkstyle + coverage
./gradlew graphviz           # Generate module state diagrams

./run_synthea                            # Generate 1 patient
./run_synthea -p 1000 -s 42 Montana      # 1000 patients, seed 42, Montana
./run_synthea -g M -a 60-65              # Males aged 60-65
./run_synthea --exporter.csv.export=true # Override config via CLI
```

### Case Studies (Python) - run from case study directory

```bash
# Sleep apnea v2 (rural underdiagnosis)
cd sleep_apnea_v2
uv run python scripts/1_generate_data.py -p 11000 -s 42
uv run python scripts/2_gen_bias.py --mask-rate 0.3
uv run python scripts/3_train_models.py
uv run python scripts/4_create_report.py

# Diabetes v2 (documentation bias)
cd diabetes_v2
uv run python scripts/1_generate_data.py -p 20000 -s 160
uv run python scripts/2_gen_bias.py
uv run python scripts/3_train_models.py
uv run python scripts/4_create_report.py
```

## Architecture

### Synthea Core (Java)

**Patient Generation Flow**: `Generator` → `Person` → `Modules` → `Exporter`

- **Generator** (`org.mitre.synthea.engine.Generator`): Orchestrates population creation with threading
- **Person** (`org.mitre.synthea.world.agents.Person`): Patient with demographics, health record, deterministic RNG
- **Modules**: JSON state machines in `src/main/resources/modules/` define disease progression
- **Exporters** (`org.mitre.synthea.export.*`): FHIR R4/STU3/DSTU2, C-CDA, CSV

**Key Directories**:
- `src/main/resources/modules/` - 106 JSON disease modules
- `src/main/resources/synthea.properties` - Main configuration
- `src/main/resources/geography/` - Census/demographics data

### Case Studies (Python)

Each case study follows a 4-script pipeline:
1. `1_generate_data.py` - Run Synthea, extract CSVs, add URBAN flag
2. `2_gen_bias.py` - Add condition flags and mask column for bias
3. `3_train_models.py` - Train GBDT models on true vs biased labels
4. `4_create_report.py` - Generate markdown report

**Output**: `output/data/` (CSVs), `output/info/` (stats), `output/report.md` (final report)

## Development Patterns

**Determinism**: Always use `person.rand()` for randomness, never `Math.random()`. This ensures reproducible simulations.

**Module Development**: JSON modules go in `src/main/resources/modules/`. Run `./gradlew graphviz` to visualize state machines.

**Code Systems**: SNOMED-CT (conditions), RxNorm (medications), LOINC (observations), CVX (immunizations)

**Timestamps**: All times in milliseconds since epoch. Use `Utilities.convertTime()` for conversions.

## Configuration

Override `synthea.properties` via:
- Command line: `--config.setting=value`
- Custom file: `-c /path/to/config.properties`
- Programmatically: `Config.set("key", "value")`

Key settings:
- `exporter.csv.export=true` - Enable CSV output
- `exporter.fhir.export=true` - Enable FHIR output
- `exporter.baseDirectory=./output` - Output location

## Testing

**Java**: JUnit 4.13.2 with Mockito. Coverage via JaCoCo in `build/reports/jacoco/test/html/`.

**Python**: Case studies use numpy, pandas, scikit-learn. Managed via uv.
