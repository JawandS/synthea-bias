# Sleep Apnea Case Study

Compare baseline demand modeling vs. a rural access bias in the sleep apnea module.

## Data Generation
Pick a state with rural counties (e.g., Maine) so the rural branch is exercised.

Baseline dataset:
```bash
./run_synthea -s 160 -p 5000 --exporter.csv.export=true --exporter.csv.append_mode=true --exporter.baseDirectory=./output_baseline Maine
```

Biased dataset (rural access bias override):
```bash
./run_synthea -s 160 -p 5000 --exporter.csv.export=true --exporter.csv.append_mode=true --exporter.baseDirectory=./output_rural_bias --module_override=/home/js/contracts/synthea-bias/config/overrides_rural_sleep_apnea.properties Maine
```

## Run The Model
```bash
scripts/.venv/bin/python3 scripts/sleep_apnea/main.py \
  --baseline output_baseline \
  --biased output_rural_bias \
  --out scripts/sleep_apnea/sleep_apnea_demand_report.md
```

## Outputs
- Markdown report: `scripts/sleep_apnea/sleep_apnea_demand_report.md`
