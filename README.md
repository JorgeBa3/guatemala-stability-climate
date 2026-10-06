# Humidity, Not Heat: medicine stability risk across Guatemalan climates

Code and results for the paper *Humidity, Not Heat: Medicine Stability Risk
Across Guatemalan Climates*.

For nine Guatemalan weather stations, the analysis computes how much faster a
medicine would degrade in the local outdoor climate than at three stability
testing conditions (25 °C/60% RH, 30 °C/65% RH and 30 °C/75% RH), using the
humidity-corrected Arrhenius equation over the published range of temperature
and humidity sensitivities.

## Reproduce

Python 3.9 or later.

```
pip install -r requirements.txt
python analysis/download_climate.py    # about 50 MB into data/climate/
python analysis/climate_kinetics.py    # writes results/*.csv
python analysis/night_ml_experiment.py # comparison of night reconstruction methods
python analysis/make_figure.py         # writes paper/fig_af.pdf and .png
```

The download step uses only the standard library and can be re-run; files
already present are skipped. Every request is logged in
`data/climate/download_log.csv`.

## Contents

| Path | What it is |
|---|---|
| `analysis/download_climate.py` | Downloads hourly station observations and reanalysis data, 2015-2025 |
| `analysis/climate_kinetics.py` | Computes coverage, climate summary, acceleration factors and the validation of the night-hour procedure |
| `analysis/night_ml_experiment.py` | Compares night-hour reconstruction methods, including a gradient-boosting model evaluated leave-one-station-out |
| `analysis/make_figure.py` | Draws Figure 1 |
| `results/station_coverage.csv` | Observed hours per station |
| `results/climate_summary.csv` | Mean temperature, mean relative humidity and mean kinetic temperature |
| `results/acceleration_factors.csv` | Acceleration factor for every station, sensitivity and reference condition, with 95% intervals |
| `results/hybrid_validation.csv` | Error of the night-hour filling procedure at fully observed stations |
| `results/night_reconstruction_hourly_error.csv` | Hourly error of each reconstruction method |
| `results/night_reconstruction_af_error.csv` | Error of each reconstruction method in the acceleration factor |
| `paper/fig_af.pdf`, `paper/fig_af.png` | Figure 1 |

Raw climate files are not stored here; the download script retrieves them.

## Data sources

- Station observations: Meteostat and its data providers, CC BY 4.0
  (https://dev.meteostat.net). Only values labelled as observations are used;
  forecast-model values are discarded.
- Reanalysis: the data was obtained from the National Aeronautics and Space
  Administration (NASA) Langley Research Center's Prediction Of Worldwide
  Energy Resources (POWER) project funded through the NASA Earth Science
  Division. Hourly data, version 2.10.2, downloaded on 2026/10/05 (UTC).

## Limitations

The records describe outdoor air at airports, not the inside of warehouses,
pharmacies or vehicles. Five of the nine stations report only during the day;
their night hours are reconstructed by anchoring reanalysis data to the evening
and morning observations, with a measured error of up to 14.4%. The humidity results apply to a product exposed to ambient air.
See the paper for the full discussion.

## Contact

Jorge Alejandro De León Batres, Universidad de San Carlos de Guatemala.
