# Notes for analysis

## Sleep Apnea
Generating data (baseline)
```bash
./run_synthea -p 25000 --exporter.csv.export=true --exporter.baseDirectory=./output_baseline
```

Generating data (baised)
```bash
./run_synthea -p 25000 --exporter.csv.export=true --exporter.baseDirectory=./output_rural_bias --module_override=/home/js/contracts/synthea-bias/config/overrides_rural_sleep_apnea.properties
```

