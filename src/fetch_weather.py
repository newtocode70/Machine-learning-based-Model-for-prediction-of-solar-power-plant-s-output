from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
URL = "https://archive-api.open-meteo.com/v1/archive"

PARAMS = {
    "latitude": 14.82,
    "longitude": 78.28,
    "start_date": "2020-05-15",
    "end_date": "2020-06-17",
    "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
    "timezone": "Asia/Kolkata",
}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(URL, params=PARAMS, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise ValueError(f"Open-Meteo response has no hourly time series: {payload}")
    if (
        abs(float(payload["latitude"]) - PARAMS["latitude"]) > 0.1
        or abs(float(payload["longitude"]) - PARAMS["longitude"]) > 0.1
        or payload.get("timezone") != PARAMS["timezone"]
    ):
        raise ValueError(
            "Open-Meteo response location/time zone does not match the requested "
            f"coordinates and timezone: {payload.get('latitude')}, "
            f"{payload.get('longitude')}, {payload.get('timezone')}."
        )
    pd.DataFrame(
        [
            ("requested_latitude", PARAMS["latitude"]),
            ("returned_latitude", payload["latitude"]),
            ("requested_longitude", PARAMS["longitude"]),
            ("returned_longitude", payload["longitude"]),
            ("requested_timezone", PARAMS["timezone"]),
            ("returned_timezone", payload["timezone"]),
        ],
        columns=["measure", "value"],
    ).to_csv(RESULTS_DIR / "openmeteo_metadata.csv", index=False)

    weather = pd.DataFrame(payload["hourly"])
    expected_columns = {
        "time",
        "shortwave_radiation",
        "temperature_2m",
        "cloud_cover",
    }
    missing_columns = expected_columns.difference(weather.columns)
    if missing_columns:
        raise ValueError(f"Open-Meteo response is missing columns: {sorted(missing_columns)}")

    weather["datetime"] = pd.to_datetime(weather.pop("time"), errors="raise")
    if weather["datetime"].duplicated().any():
        raise ValueError("Open-Meteo returned duplicate timestamps.")
    expected_times = pd.date_range(
        "2020-05-15 00:00:00", "2020-06-17 23:00:00", freq="1h"
    )
    if len(weather) != 816 or not weather["datetime"].equals(
        pd.Series(expected_times, name="datetime")
    ):
        raise ValueError(
            "Open-Meteo did not return the expected 816 hourly timestamps "
            "in Asia/Kolkata for the assignment period."
        )
    weather.to_csv(DATA_DIR / "plant1_openmeteo.csv", index=False)

    weather = weather.rename(
        columns={
            "shortwave_radiation": "public_sw_radiation",
            "temperature_2m": "public_temp2m",
        }
    )
    weather = weather[
        ["datetime", "public_sw_radiation", "public_temp2m", "cloud_cover"]
    ]

    plant = pd.read_csv(DATA_DIR / "plant1_hourly.csv", parse_dates=["datetime"])
    if len(plant) != 816:
        raise ValueError(
            f"Expected 816 prepared hourly rows, found {len(plant)}. "
            "Run src/prepare.py successfully before fetching weather."
        )
    merged = plant.merge(
        weather,
        on="datetime",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int((merged["_merge"] != "both").sum())
    merged = merged.drop(columns="_merge")
    merged.to_csv(DATA_DIR / "plant1_merged.csv", index=False)

    print(f"Open-Meteo hourly rows downloaded: {len(weather)}")
    print(f"Plant hourly rows joined: {len(merged)}")
    print(f"Plant hours without public weather: {unmatched}")
    print("\nMissing values after merge:")
    print(merged.isna().sum().to_string())

    three_days = merged[
        (merged["datetime"] >= "2020-05-15")
        & (merged["datetime"] < "2020-05-18")
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        three_days["datetime"],
        three_days["irradiation"],
        label="On-site Irradiation",
    )
    ax.plot(
        three_days["datetime"],
        three_days["public_sw_radiation"] / 1000.0,
        label="Open-Meteo Radiation / 1000",
    )
    ax.set(
        xlabel="Date and Time",
        ylabel="Radiation (kW/m²)",
        title="Three-Day Radiation Comparison",
    )
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "three_day_radiation_comparison.png", dpi=300)
    plt.close(fig)

    peak_rows = []
    for date in pd.date_range("2020-05-15", periods=3, freq="D"):
        day = merged[
            (merged["datetime"] >= date)
            & (merged["datetime"] < date + pd.Timedelta(days=1))
        ].dropna(subset=["irradiation", "public_sw_radiation"])
        if day.empty:
            raise ValueError(f"No comparable radiation data for {date.date()}.")
        onsite = day.loc[day["irradiation"].idxmax()]
        public = day.loc[day["public_sw_radiation"].idxmax()]
        peak_rows.append(
            {
                "date": date.date().isoformat(),
                "onsite_peak_time": onsite["datetime"],
                "onsite_peak_irradiation_kw_m2": onsite["irradiation"],
                "public_peak_time": public["datetime"],
                "public_peak_radiation_w_m2": public["public_sw_radiation"],
                "peak_time_matches": onsite["datetime"] == public["datetime"],
            }
        )
    peaks = pd.DataFrame(peak_rows)
    peaks.to_csv(RESULTS_DIR / "radiation_peak_check.csv", index=False)
    print("\nThree-day peak-hour check:")
    print(peaks.to_string(index=False))
    print("\nRadiation units: on-site kW/m²; Open-Meteo W/m² (divided by 1000 in plot).")


if __name__ == "__main__":
    main()
