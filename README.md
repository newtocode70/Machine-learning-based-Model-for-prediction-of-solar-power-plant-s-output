# Applied Machine Learning - Assignment 1

## Predicting Solar Power Plant Output from Weather

This project predicts hourly AC power for Plant 1 using a linear regression model implemented with NumPy. It compares:

- **Set A:** on-site irradiation, module temperature, ambient temperature, and cyclic hour features.
- **Set B:** Open-Meteo shortwave radiation, 2-metre temperature, cloud cover, and the same hour features.

The split is chronological: May 15 through June 10, 2020 for training, and June 11 through June 17, 2020 for testing. Data is never shuffled.

## Project layout

```text
app/app.py                  Streamlit predictor using the saved Set B model
data/                       Raw Kaggle CSVs and generated hourly/weather data
results/                    Metrics, figures, weights, and generated analysis
src/load_data.py            Loads all four original Kaggle CSVs
src/prepare.py              Aggregates Plant 1 power and prepares hourly data
src/eda.py                  Generates four exploratory figures
src/fetch_weather.py        Downloads, joins, and compares Open-Meteo data
src/regression.py           NumPy normal equation, batch GD, SGD, and metrics
src/train_eval.py           Time split, training, evaluation, plots, and report
```

## Setup on Windows

Install Python 3 and the dependencies:

```powershell
python.exe --version
python.exe -m pip install --user numpy pandas matplotlib requests streamlit
```

The `--user` install does not require administrator access. A virtual environment is optional.

## Publish on GitHub

1. Create an empty **public** repository on GitHub (do not initialize it with a README if this folder already has one).
2. Review the dataset license and course instructions before publishing the raw CSVs. Do not commit passwords, tokens, `.streamlit/secrets.toml`, or private information.
3. In PowerShell, run these commands from this project folder, replacing `YOUR-USERNAME` and `YOUR-REPOSITORY`:

```powershell
git init
git add .
git status
git commit -m "Prepare solar power prediction assignment"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Check `git status` before committing and verify that the intended source, README, requirements, and required project results are included. The `.gitignore` excludes virtual environments, Python caches, and Streamlit secrets. GitHub stores the repository; it does not run a Streamlit app by itself.

## Host the Streamlit app from GitHub

To make the app accessible online, push the repository first, then use Streamlit Community Cloud:

1. Sign in to Streamlit Community Cloud using GitHub and choose **Create app**.
2. Select the repository and `main` branch; set the app entry point to `app/app.py`.
3. Deploy. The app expects `results/model_weights.npz` to be present in the repository at that path.
4. Open the deployed URL and test several input combinations, including night-time/zero radiation. Check the deployment logs if it fails to load.

Do not publish this repository if the course's AI-use, public-repository, or dataset-license rules prohibit the planned contents. Ask the instructor whether publishing the source CSVs is permitted.

Run each command from the project root and confirm it exits successfully before continuing:

```powershell
Set-Location "D:\Semester 7\AML\Assignment 1"
python.exe src\load_data.py; if ($LASTEXITCODE -ne 0) { throw "load_data.py failed" }
python.exe src\prepare.py; if ($LASTEXITCODE -ne 0) { throw "prepare.py failed" }
python.exe src\eda.py; if ($LASTEXITCODE -ne 0) { throw "eda.py failed" }
python.exe src\fetch_weather.py; if ($LASTEXITCODE -ne 0) { throw "fetch_weather.py failed" }
python.exe src\train_eval.py; if ($LASTEXITCODE -ne 0) { throw "train_eval.py failed" }
python.exe -m streamlit run app\app.py
```

`fetch_weather.py` requires internet access. If a CSV is open in Excel, close it before running the preparation script; otherwise Windows may prevent the script from replacing it. Stop on any traceback and fix it before running subsequent commands, so later scripts cannot silently use stale generated data.

## Data and units

`prepare.py` sums inverter power by timestamp in the kW unit specified in the assignment, then calculates hourly means. It retains all hourly timestamps (expected: 816) in `data/plant1_hourly.csv`. Missing observations remain `NaN` and are listed in `results/missing_hourly_rows.csv` and counted in `results/data_summary.csv`; the model excludes rows missing any required feature or target. No missing target values are invented.

Open-Meteo shortwave radiation is in W/m²; on-site irradiation is in kW/m². `fetch_weather.py` divides Open-Meteo radiation by 1,000 only for the comparison figure. The model uses public radiation in W/m², matching the Streamlit input.

## What the pipeline verifies

- Plant 1’s raw files load and timestamps parse.
- Generation is summed over inverters; hourly sensor and generation values are aligned.
- The full hourly time index is retained and missing records are reported.
- Public weather uses the assignment location and date range, and joins uniquely by local timestamp.
- Train and test dates do not overlap. Scaling parameters come only from the training set.
- The normal equation, batch gradient descent, and stochastic gradient descent are compared for both feature sets.
- The selected batch-GD model must agree with the normal-equation coefficients within 0.0049 absolute difference; the chosen learning-rate check also verifies decreasing cost.
- Negative test predictions are clipped to zero before calculating all-hours and daytime RMSE.
## Generated deliverables

After running all scripts successfully, `results/` contains:

- `data_summary.csv`, `missing_hourly_rows.csv`, `openmeteo_metadata.csv`, and `radiation_peak_check.csv`
- EDA figures: `ac_power_vs_irradiation.png`, `module_vs_ambient_temp.png`, `ac_vs_dc_power.png`, `average_ac_by_hour.png`
- Optimization figures: `batch_gd_learning_rates.png`, `sgd_learning_rates.png`
- Evaluation figures/tables: `actual_vs_predicted.png`, `residuals_vs_hour.png`, `residuals_by_hour.csv`, `test_residuals.csv`, `test_predictions.csv`, `test_rmse.csv`, `learned_weights_set_a.csv`, `scaling_parameters.csv`, and `split_summary.csv`
- `model_weights.npz` for the Streamlit app
- `analysis.md`, generated from the current run’s data and metrics

Review the reported missing-row counts and RMSE values in `analysis.md`. External requirements such as the blog and LinkedIn links, public GitHub repository, commit history, and a current front-end screenshot must still be completed or verified separately.
