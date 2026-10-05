from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regression import (
    batch_gradient_descent,
    compute_cost,
    compute_rmse,
    fit_sgd,
    hypothesis,
    normal_equation,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "plant1_merged.csv"
RESULTS_DIR = ROOT / "results"
TRAIN_START = pd.Timestamp("2020-05-15")
TEST_START = pd.Timestamp("2020-06-11")
TEST_END = pd.Timestamp("2020-06-18")

FEATURES_A = [
    "irradiation",
    "module_temp",
    "ambient_temp",
    "sin_hour",
    "cos_hour",
]
FEATURES_B = [
    "public_sw_radiation",
    "public_temp2m",
    "cloud_cover",
    "sin_hour",
    "cos_hour",
]
REQUIRED_COLUMNS = list(dict.fromkeys(FEATURES_A + FEATURES_B + ["ac_power"]))
FEATURE_NAMES_A = [
    "Intercept",
    "Irradiation",
    "Module temperature",
    "Ambient temperature",
    "sin_hour",
    "cos_hour",
]


def make_design_matrix(frame, feature_columns, mean=None, std=None):
    raw = frame[feature_columns].to_numpy(dtype=float)
    if mean is None:
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    scaled = (raw - mean) / std
    return np.column_stack((np.ones(len(scaled)), scaled)), mean, std


def markdown_table(frame, formats=None):
    formats = formats or {}
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            if column in formats and pd.notna(value):
                value = formats[column](value)
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fit_converged_batch_gd(
    X, y, alpha, reference, tolerance=0.0049, max_iterations=3_000_000
):
    theta = np.zeros(X.shape[1])
    completed = 0
    check_every = 5_000
    while completed < max_iterations:
        steps = min(check_every, max_iterations - completed)
        theta, _ = batch_gradient_descent(X, y, alpha, steps, theta=theta)
        completed += steps
        difference = float(np.max(np.abs(theta - reference)))
        if difference < tolerance:
            return theta, completed, difference
    raise RuntimeError(
        f"Batch gradient descent did not match the normal equation within "
        f"{tolerance} after {max_iterations} iterations "
        f"(maximum coefficient difference {difference:.6f})."
    )


def save_learning_rate_plots(X_train, y_train):
    batch_rates = [1e-5, 1e-4, 1e-3]
    fig, ax = plt.subplots(figsize=(10, 6))
    for alpha in batch_rates:
        _, costs = batch_gradient_descent(X_train, y_train, alpha, 500)
        ax.plot(range(1, len(costs) + 1), costs, label=f"alpha = {alpha:g}")
        print(f"Batch GD alpha={alpha:g}: cost {costs[0]:.6f} -> {costs[-1]:.6f}")
    ax.set(
        xlabel="Iteration",
        ylabel="Cost J(theta)",
        title="Batch Gradient Descent Learning Rate Comparison - Set A",
    )
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "batch_gd_learning_rates.png", dpi=300)
    plt.close(fig)

    _, decrease_history = batch_gradient_descent(
        X_train, y_train, alpha=1e-3, iterations=500
    )
    increases = int(np.sum(np.diff(decrease_history) > 1e-12))
    print(
        "Batch GD cost decrease check (alpha=0.001): "
        f"largest change={np.max(np.diff(decrease_history)):.12g}, "
        f"increases={increases}/499"
    )
    if increases:
        raise AssertionError("Batch GD cost increased with the selected learning rate.")

    sgd_rates = [1e-4, 1e-3, 1e-2]
    fig, ax = plt.subplots(figsize=(10, 6))
    for alpha in sgd_rates:
        _, costs = fit_sgd(X_train, y_train, alpha, epochs=50, seed=42)
        ax.plot(range(1, len(costs) + 1), costs, label=f"alpha = {alpha:g}")
        print(f"SGD alpha={alpha:g}: cost {costs[0]:.6f} -> {costs[-1]:.6f}")
    ax.set(
        xlabel="Epoch",
        ylabel="Cost J(theta)",
        title="Stochastic Gradient Descent Learning Rate Comparison - Set A",
    )
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sgd_learning_rates.png", dpi=300)
    plt.close(fig)


def save_evaluation_plots(test, pred_a, pred_b):
    actual = test["ac_power"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, predicted, label in zip(
        axes, (pred_a, pred_b), ("Set A - Normal Equation", "Set B - Normal Equation")
    ):
        ax.scatter(actual, predicted, alpha=0.75, s=24)
        lower = min(float(actual.min()), float(predicted.min()))
        upper = max(float(actual.max()), float(predicted.max()))
        ax.plot([lower, upper], [lower, upper], "r--", label="Perfect prediction")
        ax.set(
            xlabel="Actual AC Power (kW)",
            ylabel="Predicted AC Power (kW)",
            title=label,
        )
        ax.legend()
        ax.grid(True)
    fig.suptitle("Test-Set Actual vs Predicted AC Power")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=300)
    plt.close(fig)

    residuals = actual - pred_a
    residual_frame = pd.DataFrame(
        {
            "datetime": test["datetime"].to_numpy(),
            "hour": test["datetime"].dt.hour.to_numpy(),
            "actual_ac_power_kw": actual,
            "predicted_ac_power_kw": pred_a,
            "residual_actual_minus_predicted_kw": residuals,
        }
    )
    residual_frame.to_csv(RESULTS_DIR / "test_residuals.csv", index=False)
    by_hour = residual_frame.groupby("hour")["residual_actual_minus_predicted_kw"].agg(
        ["mean", "count"]
    )
    by_hour.to_csv(RESULTS_DIR / "residuals_by_hour.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(residual_frame["hour"], residuals, alpha=0.75)
    ax.axhline(0, color="red", linestyle="--")
    ax.set(
        xlabel="Hour of Day (0-23)",
        ylabel="Residual (Actual - Predicted AC Power, kW)",
        title="Test Set Residuals vs Hour of Day (Set A Normal Equation)",
        xticks=range(24),
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "residuals_vs_hour.png", dpi=300)
    plt.close(fig)
    return residual_frame, by_hour


def write_analysis(
    data_summary,
    train_all_count,
    test_all_count,
    train,
    test,
    theta_a,
    theta_bgd,
    theta_sgd,
    rmse_frame,
    bgd_iterations,
    bgd_difference,
    bgd_b_iterations,
    bgd_b_difference,
    epochs_sgd,
    residuals_by_hour,
):
    peak_kw = float(train["ac_power"].max())
    daytime = test["irradiation"].to_numpy(dtype=float) > 0
    mean_daytime_kw = float(test.loc[daytime, "ac_power"].mean())
    rmse_a_day = float(
        rmse_frame.loc[
            (rmse_frame["Feature Set"] == "Set A")
            & (rmse_frame["Solver"] == "Normal Equation"),
            "Daytime RMSE (kW)",
        ].iloc[0]
    )
    rmse_b_day = float(
        rmse_frame.loc[
            (rmse_frame["Feature Set"] == "Set B")
            & (rmse_frame["Solver"] == "Normal Equation"),
            "Daytime RMSE (kW)",
        ].iloc[0]
    )
    extra_error = rmse_b_day - rmse_a_day
    extra_error_percent = 100 * extra_error / peak_kw if peak_kw else float("nan")

    weights = pd.DataFrame(
        {
            "Feature": FEATURE_NAMES_A,
            "Normal Equation": theta_a,
            "Batch GD": theta_bgd,
            "SGD": theta_sgd,
            "Absolute BGD - Normal": np.abs(theta_bgd - theta_a),
        }
    )
    weights.to_csv(RESULTS_DIR / "learned_weights_set_a.csv", index=False)

    training_table = pd.DataFrame(
        [
            ("Available hourly rows in training period", train_all_count),
            ("Training rows used (all required fields present)", len(train)),
            ("Training rows excluded", train_all_count - len(train)),
            ("Available hourly rows in test period", test_all_count),
            ("Test rows used (all required fields present)", len(test)),
            ("Test rows excluded", test_all_count - len(test)),
            ("Training start", train["datetime"].min().strftime("%Y-%m-%d %H:%M")),
            ("Training end", train["datetime"].max().strftime("%Y-%m-%d %H:%M")),
            ("Test start", test["datetime"].min().strftime("%Y-%m-%d %H:%M")),
            ("Test end", test["datetime"].max().strftime("%Y-%m-%d %H:%M")),
        ],
        columns=["Measure", "Value"],
    )
    training_table.to_csv(RESULTS_DIR / "split_summary.csv", index=False)

    data_table = data_summary.copy()
    openmeteo_path = DATA_PATH.parent / "plant1_openmeteo.csv"
    peak_path = RESULTS_DIR / "radiation_peak_check.csv"
    metadata_path = RESULTS_DIR / "openmeteo_metadata.csv"
    if not openmeteo_path.exists():
        raise FileNotFoundError(
            f"{openmeteo_path} is missing. Run src/fetch_weather.py before training."
        )
    if not peak_path.exists():
        raise FileNotFoundError(
            f"{peak_path} is missing. Run src/fetch_weather.py before training."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} is missing. Run src/fetch_weather.py before training."
        )
    peaks = pd.read_csv(peak_path)
    metadata = pd.read_csv(metadata_path).set_index("measure")["value"]
    extra_data_rows = pd.DataFrame(
        [
            {
                "measure": "Open-Meteo hourly points downloaded",
                "value": len(pd.read_csv(openmeteo_path)),
            },
            {
                "measure": "Three-day radiation peak times matching",
                "value": f"{int(peaks['peak_time_matches'].sum())}/{len(peaks)}",
            },
            {
                "measure": "Open-Meteo returned latitude",
                "value": metadata["returned_latitude"],
            },
            {
                "measure": "Open-Meteo returned longitude",
                "value": metadata["returned_longitude"],
            },
            {
                "measure": "Open-Meteo returned timezone",
                "value": metadata["returned_timezone"],
            },
        ]
    )
    data_table = pd.concat([data_table, extra_data_rows], ignore_index=True)
    data_table["value"] = data_table["value"].astype(str)
    residual_summary = residuals_by_hour.reset_index().rename(
        columns={
            "hour": "Hour",
            "mean": "Mean residual (kW)",
            "count": "Test observations",
        }
    )
    residual_summary["Mean residual (kW)"] = residual_summary[
        "Mean residual (kW)"
    ].map(lambda value: f"{value:.4f}")
    lines = [
        "# Assignment 1 - Results and Analysis",
        "",
        "Generated by `src/train_eval.py` from the current prepared datasets. "
        "Power metrics retain the kW unit specified for AC/DC power in the "
        "assignment. Inverter AC/DC values are summed by timestamp and averaged hourly.",
        "",
        "## Task 1: Data preparation",
        "",
        markdown_table(data_table.rename(columns={"measure": "Measure", "value": "Value"})),
        "",
        "Hourly rows containing missing source observations are retained as NaN in "
        "`data/plant1_hourly.csv` and listed in `results/missing_hourly_rows.csv`. "
        "They are excluded from model fitting/evaluation rather than being filled "
        "with invented labels or measurements.",
        "",
        "## Task 2: Exploratory plot observations",
        "",
        "- **AC power vs irradiation:** AC power rises strongly with on-site "
        "irradiation, as expected for a photovoltaic plant. Scatter around the "
        "trend reflects changing conditions, sensor variation, and hourly averaging.",
        "- **Module vs ambient temperature:** Module temperature generally rises "
        "with ambient temperature. Higher-irradiation points also tend to have "
        "warmer modules, visible in the colour gradient.",
        "- **AC vs DC power:** The two power measures have a strong positive "
        "relationship. AC output is lower than DC output, consistent with "
        "conversion and system losses.",
        "- **Average AC power by hour:** Output is near zero overnight and begins "
        "rising after sunrise. It typically peaks around midday and declines "
        "toward sunset.",
        "",
        "## Task 3: Public weather comparison",
        "",
        "Open-Meteo shortwave radiation is reported in W/m² and on-site irradiation "
        "in kW/m²; the three-day figure puts both on a common kW/m² axis. The "
        "per-day peak timestamps and the number of matching peak hours are shown "
        "in `radiation_peak_check.csv`. A one-hour or larger mismatch should be "
        "discussed as a possible difference in measurement source or alignment, "
        "not hidden by shifting the observations.",
        "",
        "## Task 4: Time split and model evaluation",
        "",
        markdown_table(training_table),
        "",
        "No rows are shuffled. Feature means and standard deviations are estimated "
        "using training rows only. Each final design matrix includes an intercept.",
        "",
        "### Test-set RMSE",
        "",
        markdown_table(
            rmse_frame,
            {
                "All Hours RMSE (kW)": lambda value: f"{value:.4f}",
                "Daytime RMSE (kW)": lambda value: f"{value:.4f}",
            },
        ),
        "",
        "Daytime is defined by on-site irradiation > 0, as specified in the brief. "
        "Predictions are clipped at zero before RMSE is calculated.",
        "",
        "### Learned Set A coefficients",
        "",
        "Features are standardized, so each non-intercept coefficient describes "
        "the target change associated with a one-standard-deviation change in its "
        "feature, holding other features fixed.",
        "",
        markdown_table(
            weights,
            {
                "Normal Equation": lambda value: f"{value:.6f}",
                "Batch GD": lambda value: f"{value:.6f}",
                "SGD": lambda value: f"{value:.6f}",
                "Absolute BGD - Normal": lambda value: f"{value:.6f}",
            },
        ),
        "",
        f"Irradiation has the largest Set A coefficient "
        f"({theta_a[1]:.4f}), and its positive sign agrees with the expected "
        "physical relationship between sunlight and solar output. Temperature "
        "and hour-feature signs are conditional associations in this linear model "
        "and should not be interpreted as causal effects.",
        "",
        "### Task 5: Interpretation",
        "",
        f"- Training-period peak hourly AC output: **{peak_kw:.3f} kW**.",
        f"- Mean daytime test output: **{mean_daytime_kw:.3f} kW** "
        f"({100 * mean_daytime_kw / peak_kw if peak_kw else float('nan'):.1f}% "
        "of the training peak).",
        f"- Set B Normal Equation daytime RMSE exceeds Set A by "
        f"**{extra_error:.3f} kW**, equal to **{extra_error_percent:.1f}%** "
        "of the training peak. Set A is the more accurate feature set on this "
        "held-out period; the public-weather error should be judged against the "
        "application's tolerance, not just against Set A.",
        f"- Batch GD used **{bgd_iterations:,} iterations** at alpha=0.001 and "
        f"finished within **{bgd_difference:.6f}** maximum absolute coefficient "
        "difference of the Normal Equation solution (acceptance threshold: 0.0049).",
        f"- Set B Batch GD used **{bgd_b_iterations:,} iterations** at alpha=0.001 "
        "and finished "
        f"within **{bgd_b_difference:.6f}** maximum coefficient difference. "
        f"SGD used **{epochs_sgd} epochs** at alpha=0.01 for the final comparison.",
        "- The Normal Equation computes a closed-form least-squares solution "
        "without iterative epochs. Compare its coefficients with Batch GD and SGD "
        "in the table above; exact agreement is specifically required for Batch GD.",
        "- Batch GD updates from the full dataset each iteration and gives a "
        "smooth cost curve; it is practical for this small dataset but expensive "
        "for very large data.",
        "- SGD updates one example at a time, so it can make progress on large or "
        "streaming datasets but its cost curve is noisier. The deterministic seed "
        "makes the reported run repeatable.",
        "- The BGD and SGD learning-rate plots are saved as "
        "`batch_gd_learning_rates.png` and `sgd_learning_rates.png`.",
        "- Within the tested rate ranges, smaller rates make less progress over "
        "the fixed plotting budget and the largest tested rates make faster "
        "progress. If all displayed costs decrease, none of the tested rates is "
        "too large; do not describe a stable rate as divergent.",
        "- The test residual plot and per-hour mean residual table are saved as "
        "`residuals_vs_hour.png` and `residuals_by_hour.csv`. Residual structure "
        "by hour signals systematic time-dependent errors that should be discussed "
        "from the plotted values.",
        "",
        "## Required figures",
        "",
        "Exploratory plots: `ac_power_vs_irradiation.png`, "
        "`module_vs_ambient_temp.png`, `ac_vs_dc_power.png`, and "
        "`average_ac_by_hour.png`. The model comparison figure is "
        "`actual_vs_predicted.png`.",
        "",
        "### Residuals by hour",
        "",
        markdown_table(residual_summary),
        "",
        "Residuals are actual minus predicted. Positive means indicate "
        "underprediction; negative means indicate overprediction. Use this table "
        "and `residuals_vs_hour.png` to assess time-dependent bias.",
        "",
        "## Remaining manual submission items",
        "",
        "Add the required blog/LinkedIn links and front-end screenshot to the "
        "submission package, and verify the public repository URL and commit "
        "history. Those external items cannot be generated by this local pipeline.",
    ]
    (RESULTS_DIR / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    if len(df) != 816:
        raise ValueError(
            f"Expected 816 hourly rows in {DATA_PATH}, found {len(df)}. "
            "Run src/prepare.py and src/fetch_weather.py successfully first."
        )
    hourly_path = DATA_PATH.parent / "plant1_hourly.csv"
    if DATA_PATH.stat().st_mtime_ns < hourly_path.stat().st_mtime_ns:
        raise RuntimeError(
            "The merged dataset is older than the prepared hourly data. "
            "Run src/fetch_weather.py again before training."
        )
    if df["datetime"].duplicated().any():
        raise ValueError("Merged dataset has duplicate hourly timestamps.")
    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("Merged dataset must be sorted chronologically.")

    df["hour"] = df["datetime"].dt.hour
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

    in_train_period = (df["datetime"] >= TRAIN_START) & (df["datetime"] < TEST_START)
    in_test_period = (df["datetime"] >= TEST_START) & (df["datetime"] < TEST_END)
    train_period = df.loc[in_train_period].copy()
    test_period = df.loc[in_test_period].copy()
    if train_period.empty or test_period.empty:
        raise ValueError("Training or test date range has no rows.")

    train_valid = train_period.dropna(subset=REQUIRED_COLUMNS).copy()
    test_valid = test_period.dropna(subset=REQUIRED_COLUMNS).copy()
    if train_valid.empty or test_valid.empty:
        raise ValueError("No complete rows remain in the training or test period.")

    train_finite = np.isfinite(train_valid[REQUIRED_COLUMNS].to_numpy(dtype=float)).all()
    test_finite = np.isfinite(test_valid[REQUIRED_COLUMNS].to_numpy(dtype=float)).all()
    if not train_finite or not test_finite:
        raise ValueError("Training/test features contain non-finite values.")
    if not (train_valid["datetime"].max() < test_valid["datetime"].min()):
        raise AssertionError("Training and test periods overlap.")

    print(f"Hourly rows in training period: {len(train_period)}")
    print(f"Training rows used: {len(train_valid)}")
    print(f"Training rows excluded for missing values: {len(train_period) - len(train_valid)}")
    print(f"Hourly rows in test period: {len(test_period)}")
    print(f"Test rows used: {len(test_valid)}")
    print(f"Test rows excluded for missing values: {len(test_period) - len(test_valid)}")
    print(
        f"Train dates: {train_valid['datetime'].min()} to "
        f"{train_valid['datetime'].max()}"
    )
    print(
        f"Test dates: {test_valid['datetime'].min()} to "
        f"{test_valid['datetime'].max()}"
    )

    X_train_a, mean_a, std_a = make_design_matrix(train_valid, FEATURES_A)
    X_test_a, _, _ = make_design_matrix(test_valid, FEATURES_A, mean_a, std_a)
    X_train_b, mean_b, std_b = make_design_matrix(train_valid, FEATURES_B)
    X_test_b, _, _ = make_design_matrix(test_valid, FEATURES_B, mean_b, std_b)
    scaling_parameters = pd.DataFrame(
        {
            "Feature Set": ["Set A"] * len(FEATURES_A) + ["Set B"] * len(FEATURES_B),
            "Feature": FEATURES_A + FEATURES_B,
            "Training mean": np.concatenate((mean_a, mean_b)),
            "Training std": np.concatenate((std_a, std_b)),
        }
    )
    scaling_parameters.to_csv(RESULTS_DIR / "scaling_parameters.csv", index=False)
    y_train = train_valid["ac_power"].to_numpy(dtype=float)
    y_test = test_valid["ac_power"].to_numpy(dtype=float)

    print(f"\nSet A design matrices: train {X_train_a.shape}; test {X_test_a.shape}")
    print(f"Set B design matrices: train {X_train_b.shape}; test {X_test_b.shape}")
    print("Feature means/stds are calculated on training data only.")

    theta_a_ne = normal_equation(X_train_a, y_train)
    theta_b_ne = normal_equation(X_train_b, y_train)
    print("\nNormal-equation training costs:")
    print(f"Set A: {compute_cost(X_train_a, y_train, theta_a_ne):.6f}")
    print(f"Set B: {compute_cost(X_train_b, y_train, theta_b_ne):.6f}")

    save_learning_rate_plots(X_train_a, y_train)

    alpha_bgd = 0.001
    theta_a_bgd, bgd_iterations, bgd_difference = fit_converged_batch_gd(
        X_train_a, y_train, alpha_bgd, theta_a_ne
    )
    theta_b_bgd, bgd_b_iterations, bgd_b_difference = fit_converged_batch_gd(
        X_train_b, y_train, alpha_bgd, theta_b_ne
    )
    alpha_sgd = 0.01
    epochs_sgd = 100
    theta_a_sgd, _ = fit_sgd(X_train_a, y_train, alpha_sgd, epochs_sgd, seed=42)
    theta_b_sgd, _ = fit_sgd(X_train_b, y_train, alpha_sgd, epochs_sgd, seed=42)

    print("\nConvergence agreement with the normal equation:")
    print(
        f"Set A BGD: {bgd_iterations:,} iterations, "
        f"max |theta difference|={bgd_difference:.6f}"
    )
    print(
        f"Set B BGD: {bgd_b_iterations:,} iterations, "
        f"max |theta difference|={bgd_b_difference:.6f}"
    )

    daytime_mask = test_valid["irradiation"].to_numpy(dtype=float) > 0
    if not daytime_mask.any():
        raise ValueError("The held-out test set contains no daytime observations.")

    solver_models = [
        ("Normal Equation", theta_a_ne, theta_b_ne),
        ("Batch GD", theta_a_bgd, theta_b_bgd),
        ("SGD", theta_a_sgd, theta_b_sgd),
    ]
    rows = []
    predictions = {}
    for name, theta_a, theta_b in solver_models:
        pred_a = np.maximum(hypothesis(X_test_a, theta_a), 0)
        pred_b = np.maximum(hypothesis(X_test_b, theta_b), 0)
        predictions[name] = (pred_a, pred_b)
        rows.append(
            {
                "Feature Set": "Set A",
                "Solver": name,
                "All Hours RMSE (kW)": compute_rmse(y_test, pred_a),
                "Daytime RMSE (kW)": compute_rmse(y_test[daytime_mask], pred_a[daytime_mask]),
            }
        )
        rows.append(
            {
                "Feature Set": "Set B",
                "Solver": name,
                "All Hours RMSE (kW)": compute_rmse(y_test, pred_b),
                "Daytime RMSE (kW)": compute_rmse(y_test[daytime_mask], pred_b[daytime_mask]),
            }
        )
    rmse_frame = pd.DataFrame(rows)
    print("\nTABLE 2: TEST RMSE (kW)")
    print(rmse_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    rmse_frame.to_csv(RESULTS_DIR / "test_rmse.csv", index=False)

    theta_frame = pd.DataFrame(
        {
            "Feature": FEATURE_NAMES_A,
            "Normal Equation": theta_a_ne,
            "Batch GD": theta_a_bgd,
            "SGD": theta_a_sgd,
            "Absolute BGD - Normal": np.abs(theta_a_bgd - theta_a_ne),
        }
    )
    print("\nTABLE 3: LEARNED THETA COMPARISON (SET A)")
    print(theta_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    pred_a_ne, pred_b_ne = predictions["Normal Equation"]
    prediction_table = pd.DataFrame(
        {
            "datetime": test_valid["datetime"].to_numpy(),
            "actual_ac_power_kw": y_test,
            "predicted_set_a_normal_kw": pred_a_ne,
            "predicted_set_b_normal_kw": pred_b_ne,
        }
    )
    prediction_table.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    _, residuals_by_hour = save_evaluation_plots(test_valid, pred_a_ne, pred_b_ne)

    np.savez_compressed(
        RESULTS_DIR / "model_weights.npz",
        theta_B=theta_b_ne,
        mean_B=mean_b,
        std_B=std_b,
        target_unit=np.array("kW"),
    )

    data_summary_path = RESULTS_DIR / "data_summary.csv"
    if not data_summary_path.exists():
        raise FileNotFoundError(
            f"{data_summary_path} is missing. Run src/prepare.py before training."
        )
    data_summary = pd.read_csv(data_summary_path)
    write_analysis(
        data_summary,
        len(train_period),
        len(test_period),
        train_valid,
        test_valid,
        theta_a_ne,
        theta_a_bgd,
        theta_a_sgd,
        rmse_frame,
        bgd_iterations,
        bgd_difference,
        bgd_b_iterations,
        bgd_b_difference,
        epochs_sgd,
        residuals_by_hour,
    )
    print("\nSaved model weights, tables, plots, predictions, and results/analysis.md.")


if __name__ == "__main__":
    main()
