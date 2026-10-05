from pathlib import Path

import pandas as pd

from load_data import load_raw


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write {path}. Close it in Excel or another program, "
            "then rerun this script."
        ) from exc


def prepare_hourly_data(raw):
    generation = raw["gen1"].copy()
    sensor = raw["sensor1"].copy()

    generation["datetime"] = pd.to_datetime(
        generation["DATE_TIME"], format="%d-%m-%Y %H:%M", errors="raise"
    )
    sensor["datetime"] = pd.to_datetime(
        sensor["DATE_TIME"], format="%Y-%m-%d %H:%M:%S", errors="raise"
    )

    generation_15m = generation.groupby("datetime", as_index=True).agg(
        {"AC_POWER": "sum", "DC_POWER": "sum"}
    )
    sensor_15m = sensor.groupby("datetime", as_index=True).agg(
        {
            "AMBIENT_TEMPERATURE": "mean",
            "MODULE_TEMPERATURE": "mean",
            "IRRADIATION": "mean",
        }
    )
    generation_times = set(generation["datetime"].astype("int64"))
    sensor_times = set(sensor["datetime"].astype("int64"))

    generation_hourly = generation_15m.resample("1h").mean()
    sensor_hourly = sensor_15m.resample("1h").mean()

    hourly = generation_hourly.join(sensor_hourly, how="outer")
    start = min(generation["datetime"].min(), sensor["datetime"].min()).floor("h")
    end = max(generation["datetime"].max(), sensor["datetime"].max()).floor("h")
    full_hours = pd.date_range(start=start, end=end, freq="1h", name="datetime")
    if (
        len(full_hours) != 816
        or start != pd.Timestamp("2020-05-15 00:00:00")
        or end != pd.Timestamp("2020-06-17 23:00:00")
    ):
        raise ValueError(
            f"Expected 816 hourly timestamps for the assignment date range, "
            f"found {len(full_hours)} ({start} through {end})."
        )
    hourly = hourly.reindex(full_hours).rename(
        columns={
            "AC_POWER": "ac_power",
            "DC_POWER": "dc_power",
            "AMBIENT_TEMPERATURE": "ambient_temp",
            "MODULE_TEMPERATURE": "module_temp",
            "IRRADIATION": "irradiation",
        }
    )
    hourly.index.name = "datetime"
    hourly = hourly.reset_index()

    summary = pd.DataFrame(
        [
            ("raw_generation_rows_plant_1", len(generation)),
            ("generation_timestamps_plant_1", generation["datetime"].nunique()),
            ("raw_sensor_rows_plant_1", len(sensor)),
            ("sensor_timestamps_plant_1", sensor["datetime"].nunique()),
            ("timestamps_only_in_generation", len(generation_times - sensor_times)),
            ("timestamps_only_in_sensor", len(sensor_times - generation_times)),
            ("timestamps_present_in_only_one_file", len(generation_times ^ sensor_times)),
            ("hourly_rows_expected", len(full_hours)),
            ("hourly_rows_with_any_missing_value", int(hourly.isna().any(axis=1).sum())),
            ("hourly_rows_complete", int(hourly.notna().all(axis=1).sum())),
            ("missing_cells", int(hourly.isna().sum().sum())),
        ],
        columns=["measure", "value"],
    )
    return hourly, summary


def main():
    raw = load_raw(DATA_DIR)
    hourly, summary = prepare_hourly_data(raw)

    save_csv(hourly, DATA_DIR / "plant1_hourly.csv")
    save_csv(summary, RESULTS_DIR / "data_summary.csv")
    missing_hours = hourly.loc[hourly.isna().any(axis=1), ["datetime"]]
    save_csv(missing_hours, RESULTS_DIR / "missing_hourly_rows.csv")

    print("Plant 1 hourly dataset saved to data/plant1_hourly.csv")
    print("Power columns retain the source unit kW specified by the assignment.")
    print(summary.to_string(index=False))
    print("\nMissing values by column:")
    print(hourly.isna().sum().to_string())
    print(
        "\nMissing hourly observations are retained as NaN for transparent "
        "reporting. Training/evaluation excludes rows missing required fields."
    )


if __name__ == "__main__":
    main()
