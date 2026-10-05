from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "plant1_hourly.csv"
RESULTS_DIR = ROOT / "results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    complete = df.dropna(
        subset=[
            "ac_power",
            "dc_power",
            "ambient_temp",
            "module_temp",
            "irradiation",
        ]
    )
    print(f"Plotting {len(complete)} complete hourly observations.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(complete["irradiation"], complete["ac_power"], s=10)
    ax.set(
        xlabel="Irradiation (kW/m²)",
        ylabel="AC Power (kW)",
        title="AC Power vs Irradiation",
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "ac_power_vs_irradiation.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    points = ax.scatter(
        complete["ambient_temp"],
        complete["module_temp"],
        c=complete["irradiation"],
        s=10,
    )
    ax.set(
        xlabel="Ambient Temperature (°C)",
        ylabel="Module Temperature (°C)",
        title="Module Temperature vs Ambient Temperature",
    )
    fig.colorbar(points, ax=ax, label="Irradiation (kW/m²)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "module_vs_ambient_temp.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(complete["dc_power"], complete["ac_power"], s=10)
    ax.set(
        xlabel="DC Power (kW)",
        ylabel="AC Power (kW)",
        title="AC Power vs DC Power",
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "ac_vs_dc_power.png", dpi=300)
    plt.close(fig)

    complete = complete.copy()
    complete["hour"] = complete["datetime"].dt.hour
    hourly_ac = complete.groupby("hour")["ac_power"].mean().reindex(range(24))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hourly_ac.index, hourly_ac.values, marker="o")
    ax.set(
        xlabel="Hour of Day",
        ylabel="Average AC Power (kW)",
        title="Average AC Power by Hour of Day",
        xticks=range(24),
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "average_ac_by_hour.png", dpi=300)
    plt.close(fig)

    print("\nObservations:")
    print("1. AC power generally increases with on-site irradiation.")
    print("2. Module temperature generally rises with ambient temperature and irradiation.")
    print("3. AC output tracks DC output, with conversion/system losses.")
    print("4. Power is low overnight and generally peaks around midday.")


if __name__ == "__main__":
    main()
