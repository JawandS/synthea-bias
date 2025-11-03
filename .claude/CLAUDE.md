# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthea is a Synthetic Patient Population Simulator that generates realistic (but not real) patient data and associated health records. It simulates patients from birth to death with realistic medical histories, encounters, conditions, medications, and more.

**Requirements**: Java JDK 11 or newer (LTS versions 11 or 17 recommended)

## Build and Test Commands

### Building
```bash
# Build the project
./gradlew build

# Build and run all checks and tests
./gradlew build check test

# Clean build artifacts
./gradlew clean

# Create standalone JAR with dependencies
./gradlew shadowJar
```

### Testing
```bash
# Run all tests
./gradlew test

# Run a specific test class
./gradlew test --tests "org.mitre.synthea.engine.ModuleTest"

# Run a specific test method
./gradlew test --tests "org.mitre.synthea.engine.ModuleTest.testSpecificMethod"

# Run tests with code coverage
./gradlew test jacocoTestReport
```

### Code Quality
```bash
# Run checkstyle
./gradlew checkstyle

# Run all verification tasks (tests + checkstyle + jacoco)
./gradlew check
```

### Generating Patients
```bash
# Generate a single patient
./run_synthea

# Generate specific population
./run_synthea -p 1000                    # 1000 patients
./run_synthea -s 12345                   # with specific seed
./run_synthea Massachusetts              # in Massachusetts
./run_synthea Alaska Juneau              # in Juneau, Alaska
./run_synthea -g M -a 60-65              # males aged 60-65

# Override config settings via command line
./run_synthea -p 10 --exporter.fhir.export=true
./run_synthea --exporter.baseDirectory="./output_tx/" Texas
```

### Utility Tasks
```bash
# Generate GraphViz visualizations of modules
./gradlew graphviz

# Generate list of simulated concepts
./gradlew concepts

# Generate list of patient attributes
./gradlew attributes

# Test physiology simulation
./gradlew physiology
```

## Architecture

### Core Patient Generation Flow

1. **Generator** (`org.mitre.synthea.engine.Generator`): Main orchestrator that creates populations
   - Manages threading and parallel generation
   - Configurable via `GeneratorOptions`
   - Processes patients through timesteps (default: 7 days = 604800000ms)

2. **Person** (`org.mitre.synthea.world.agents.Person`): Represents a simulated patient
   - Stores demographics, vital signs, and health history
   - Contains `HealthRecord` with all medical events
   - Implements `RandomNumberGenerator` for deterministic randomness per person

3. **Module System**: Two types of modules drive patient simulation:
   - **Java Modules**: Hard-coded modules like `LifecycleModule`, `EncounterModule`, `CardiovascularDiseaseModule`
   - **Generic Modules**: JSON-based state machines in `src/main/resources/modules/`
     - Define disease progression as state transitions
     - Use logic conditions, delays, and transitions
     - Modular and composable (supports submodules)

### Module State Machine

Modules are implemented as state machines where each state can be:
- **Initial**: Entry point for the module
- **Terminal**: Exit point for the module
- **Delay**: Wait for a period of time
- **Guard**: Conditional branching
- **SetAttribute**: Set patient attributes
- **Encounter**: Trigger a healthcare encounter
- **ConditionOnset/ConditionEnd**: Start/stop conditions
- **MedicationOrder/MedicationEnd**: Prescribe/stop medications
- **Procedure**: Perform a medical procedure
- **Observation**: Record vital signs or lab results
- **Death**: End patient life

States connect via **Transitions** which can be:
- Direct (immediate)
- Distributed (probabilistic)
- Conditional (logic-based)
- Complex (table lookups)

### Export System

The export system supports multiple formats via the `Exporter` abstraction:

- **FHIR**: R4 (default), STU3, DSTU2
  - Enable: Set `exporter.fhir.export = true`
  - Bulk FHIR (ndjson): Set `exporter.fhir.bulk_data = true`
  - Transaction bundles for referential integrity

- **C-CDA**: Enable via `exporter.ccda.export = true`

- **CSV**: Enable via `exporter.csv.export = true`
  - Outputs multiple CSV files (patients, encounters, conditions, medications, etc.)
  - Configurable file inclusion/exclusion

- **Custom Exporters**: Implement `PatientExporter` or `PostCompletionExporter`
  - Loaded via Java ServiceLoader when `exporter.enable_custom_exporters = true`

**Key Export Classes**:
- `FhirR4.java`: FHIR R4 conversion logic
- `CSVExporter.java`: CSV export implementation
- `CCDAExporter.java`: C-CDA document generation
- `Flexporter`: Customizable FHIR transformation framework

### World Model

**Geography & Demographics** (`org.mitre.synthea.world.geography`):
- `Location`: Represents cities/states with demographic data
- `Demographics`: Loads census data for realistic population distribution

**Agents** (`org.mitre.synthea.world.agents`):
- `Person`: The patient being simulated
- `Provider`: Healthcare facilities (hospitals, clinics, etc.)
- `Payer`: Insurance companies and plans
- `Clinician`: Healthcare providers

**Concepts** (`org.mitre.synthea.world.concepts`):
- `HealthRecord`: Complete medical record for a person
  - Encounters, Conditions, Medications, Procedures, Observations, etc.
- `Costs`: Cost modeling for procedures, medications, encounters
- `Claim`: Insurance claims and reimbursement

**Health Insurance System**:
- Complex insurance eligibility and enrollment logic
- Multiple payer types: Private, Medicare, Medicaid, Dual Eligible
- Plan selection behaviors: best_rates, random, priority
- Configurable via `generate.payers.*` settings

## Configuration

Main configuration file: `src/main/resources/synthea.properties`

Key configuration areas:
- **Export formats**: Enable/disable different output formats
- **Demographics**: Population distribution, geographic data
- **Modules**: Which modules to run, module-specific parameters
- **Costs**: Default costs for procedures, medications, encounters
- **Providers**: Provider selection behavior, search distance
- **Payers**: Insurance plan selection, adjustment behavior
- **Physiology**: Enable/disable physiology simulations
- **Lifecycle**: Quit smoking/alcoholism rates, adherence, death parameters

Configuration can be overridden via:
- Command line: `--config.setting=value`
- Custom config file: `-c /path/to/config.properties`
- Programmatically: `Config.set("key", "value")`

## Key File Locations

- **Modules**: `src/main/resources/modules/` - JSON state machine definitions
- **Demographics**: `src/main/resources/geography/` - Census and location data
- **Providers**: `src/main/resources/providers/` - Hospital and facility data
- **Payers**: `src/main/resources/payers/` - Insurance company and plan data
- **Costs**: `src/main/resources/costs/` - Pricing data
- **Lookup Tables**: `src/main/resources/modules/lookup_tables/` - Module data tables
- **Output**: `./output/` - Generated patient records (configurable)

## Testing Notes

- Test framework: JUnit 4.13.2
- Mocking: Mockito + PowerMock
- Test utilities in `TestHelper.java` provide common setup methods
- Tests use deterministic seeds for reproducibility
- Coverage reports generated via JaCoCo in `build/reports/jacoco/test/html/`

## Important Development Patterns

1. **Module Development**: When creating new modules:
   - JSON modules go in `src/main/resources/modules/`
   - Use the Generic Module Framework (GMF) version 2.0
   - Follow existing module patterns for state machines
   - Use `./gradlew graphviz` to visualize module flow

2. **Random Number Generation**: Always use person-specific RNG
   - Each Person implements `RandomNumberGenerator`
   - Use `person.rand()` for deterministic, reproducible randomness
   - Never use `Math.random()` or `new Random()`

3. **Exporter Implementation**:
   - Patient-level exporters implement `PatientExporter`
   - Post-generation exporters implement `PostCompletionExporter`
   - Register via ServiceLoader in `META-INF/services/`

4. **Costs and Claims**:
   - Use cost data from `src/main/resources/costs/`
   - Claims are generated automatically based on encounters
   - Payer adjudication follows configured adjustment behavior

5. **Timestamps**:
   - All times are in milliseconds since epoch
   - Timestep configurable via `generate.timestep` (default: 1 week)
   - Use `Utilities.convertTime()` for time unit conversions

6. **Code Systems**:
   - SNOMED-CT for conditions
   - RxNorm for medications
   - LOINC for observations
   - CVX for immunizations
   - Use `HealthRecord.Code` class for all clinical codes
