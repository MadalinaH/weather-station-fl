"""
standardise_data.py
===================
Phase 1

PURPOSE
-------
This script does NOT fetch data from the internet.  The raw CSVs were
manually downloaded from the FMI (Finnish Meteorological Institute) open
data portal:

    https://en.ilmatieteenlaitos.fi/download-observations

HOW THE RAW DATA WAS DOWNLOADED
--------------------------------
For each of the 9 stations, the portal was configured as follows:

  Tab        : Hourly observations
  Variables  : Average temperature, Maximum temperature,
               Minimum temperature, Wind speed
  Time period: 01/01/2022 - 31/12/2024
  Time zone  : UTC
  Format     : CSV

The portal returns one CSV per station with hourly rows and this header:

  "Observation station","Year","Month","Day","Time [UTC]",
  "Wind speed [m/s]","Average temperature [°C]",
  "Maximum temperature [°C]","Minimum temperature [°C]"

WHY HOURLY AND NOT DAILY?
--------------------------
The FMI portal's "Daily observations" tab does not include wind speed -
only temperature and precipitation variables are available at daily
resolution.  To get all four variables we need (t2m, tmin, tmax, ws_10min)
in a single download, we use hourly data and aggregate ourselves.

WHAT THIS SCRIPT DOES
----------------------
For each station CSV in fl_project/data/:

  1. Parse the Year/Month/Day/Time columns into a proper UTC datetime.
  2. Rename FMI column headers to our internal names.
  3. Aggregate hourly readings to one row per UTC calendar day:
       t2m      = mean(Average temperature)  [°C]
       tmin     = min (Minimum temperature)  [°C]
       tmax     = max (Maximum temperature)  [°C]
       ws_10min = mean(Wind speed)           [m/s]
  4. Reindex to the full date range 2022-01-01 → 2024-12-31 so every
     calendar day has a row (missing days become NaN rows).
  5. Overwrite the raw CSV with the clean daily version.

KNOWN DATA QUALITY ISSUES
--------------------------
  - Tampere Härmälä    : ws_10min entirely absent (station has no anemometer).
  - Inari Saariselkä  : ws_10min entirely absent (same reason).
  - Turku Artukainen  : 22 days of missing ws_10min (instrument gaps).
  - Jyväskylä Airport : 1 day missing across all variables.
  - Sodankylä         : 2 days missing across all variables.

  Missing values are left as NaN here and handled in prepare_data.py
  (rows with any NaN are dropped per station before model training).

OUTPUT
------
  fl_project/data/<station>.csv        one clean daily CSV per station
  fl_project/data/missing_summary.csv  NaN counts per station per column

Usage:
    python standardise_data.py
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Station registry
# ---------------------------------------------------------------------------

# Maps our internal project label → input CSV filename (in DATA_DIR).
STATIONS = {
    "Helsinki Kaisaniemi"       : "helsinki.csv",
    "Turku Artukainen"          : "turku.csv",
    "Oulu Vihreäsaari"         : "oulu.csv",
    "Tampere Härmälä"          : "tampere.csv",
    "Jyväskylä Airport"        : "jyvaskyla.csv",
    "Kuopio Maaninka"           : "kuopio.csv",
    "Rovaniemi Apukka"          : "rovaniemi.csv",
    "Sodankylä"                 : "sodankyla.csv",
    "Inari Saariselkä"         : "inari.csv",
    "Hanko Tulliniemi"          : "hanko.csv",
    "Kajaani Airport"           : "kajaani.csv",
    "Kittilä Airport"           : "kittila.csv",
    "Muonio Oustajärvi"        : "muonio.csv",
    "Pelkosenniemi Pyhätunturi" : "pelkosenniemi.csv",
    "Raahe Nahkiainen"          : "raahe.csv",
}

# FMI portal column names → our internal names.
# The portal wraps everything in quotes and uses units in the header.
COLUMN_RENAME = {
    "Wind speed [m/s]"          : "ws_10min",
    "Average temperature [°C]"  : "avg_temp",
    "Maximum temperature [°C]"  : "tmax_raw",
    "Minimum temperature [°C]"  : "tmin_raw",
}

# Full project date range - used to reindex so every day has a row.
DATE_START = "2022-01-01"
DATE_END   = "2024-12-31"

# Final CSV column order.
REQUIRED_COLUMNS = ["date", "t2m", "tmin", "tmax", "ws_10min"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Load and clean one station's hourly CSV
# ---------------------------------------------------------------------------

def load_hourly(csv_path: str) -> pd.DataFrame:
    """
    Read one FMI portal hourly CSV and return a tidy DataFrame.

    Steps:
      1. Read the CSV (quoted strings, comma-separated).
      2. Build a proper UTC datetime column from Year/Month/Day/Time columns.
      3. Rename FMI headers to internal column names.
      4. Coerce all measurement columns to float (FMI uses "" for missing).

    Parameters
    ----------
    csv_path : absolute path to the downloaded CSV file

    Returns
    -------
    pd.DataFrame with columns [datetime_utc, avg_temp, tmin_raw, tmax_raw, ws_10min]
    """
    df = pd.read_csv(csv_path, dtype=str)

    # Strip any accidental whitespace from column names.
    df.columns = df.columns.str.strip()

    # Build a datetime from the separate Year / Month / Day / Time [UTC] cols.
    # Time column looks like "01:00", "14:00" etc.
    df["datetime_utc"] = pd.to_datetime(
        df["Year"].str.strip() + "-" +
        df["Month"].str.strip().str.zfill(2) + "-" +
        df["Day"].str.strip().str.zfill(2) + " " +
        df["Time [UTC]"].str.strip(),
        format="%Y-%m-%d %H:%M",
        utc=True,
    )

    # Rename measurement columns.
    df = df.rename(columns=COLUMN_RENAME)

    # Keep only the columns we need.
    keep = ["datetime_utc"] + list(COLUMN_RENAME.values())
    df = df[[c for c in keep if c in df.columns]]

    # Convert measurements to float; FMI uses empty string for missing values.
    for col in COLUMN_RENAME.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Aggregate hourly → daily
# ---------------------------------------------------------------------------

def aggregate_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample hourly observations to one row per UTC calendar day.

    Aggregation rules:
        t2m      = mean of avg_temp  (hourly average temperatures → daily mean)
        tmin     = min  of tmin_raw  (hourly minimum temperatures → daily min)
        tmax     = max  of tmax_raw  (hourly maximum temperatures → daily max)
        ws_10min = mean of ws_10min  (hourly mean wind speeds → daily mean)

    Days where all hourly values are NaN produce a NaN row; they are kept so
    the date index stays contiguous (missing days handled in prepare_data.py).

    Parameters
    ----------
    hourly_df : output of load_hourly()

    Returns
    -------
    pd.DataFrame with columns [date, t2m, tmin, tmax, ws_10min]
    """
    df = hourly_df.copy()
    df["date"] = df["datetime_utc"].dt.date

    agg_dict = {}
    if "avg_temp"  in df.columns: agg_dict["t2m"]      = ("avg_temp",  "mean")
    if "tmin_raw"  in df.columns: agg_dict["tmin"]     = ("tmin_raw",  "min")
    if "tmax_raw"  in df.columns: agg_dict["tmax"]     = ("tmax_raw",  "max")
    if "ws_10min"  in df.columns: agg_dict["ws_10min"] = ("ws_10min",  "mean")

    daily = df.groupby("date").agg(**agg_dict).reset_index()

    # Reindex so every calendar day in [DATE_START, DATE_END] has a row.
    full_range = pd.DataFrame({
        "date": pd.date_range(DATE_START, DATE_END, freq="D").date
    })
    daily = full_range.merge(daily, on="date", how="left")

    # Ensure all required columns exist even if source was missing a variable.
    for col in ["t2m", "tmin", "tmax", "ws_10min"]:
        if col not in daily.columns:
            daily[col] = np.nan

    # Round to 2 decimal places for readable CSVs.
    for col in ["t2m", "tmin", "tmax", "ws_10min"]:
        daily[col] = daily[col].round(2)

    return daily[REQUIRED_COLUMNS]


# ---------------------------------------------------------------------------
# Missing-value summary
# ---------------------------------------------------------------------------

def build_missing_summary(station_data: dict) -> pd.DataFrame:
    """
    Return a DataFrame summarising NaN counts per column per station.

    Parameters
    ----------
    station_data : dict mapping project_label → daily DataFrame

    Returns
    -------
    pd.DataFrame with columns [station, column, missing_count, missing_pct]
    """
    rows = []
    for station, df in station_data.items():
        for col in ["t2m", "tmin", "tmax", "ws_10min"]:
            n_missing = int(df[col].isna().sum()) if col in df.columns else len(df)
            pct = round(100.0 * n_missing / max(len(df), 1), 2)
            rows.append({"station": station, "column": col,
                         "missing_count": n_missing, "missing_pct": pct})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Orchestrate Phase 1:
      1. For each station, load its hourly CSV and aggregate to daily stats.
      2. Save the daily CSV to data/<station>_daily.csv.
      3. Print and save a missing-value summary.
    """
    print("=" * 65)
    print("  FL Project - Phase 1: Standardising FMI portal CSVs")
    print(f"  Date range  : {DATE_START} → {DATE_END}")
    print(f"  Stations    : {len(STATIONS)}")
    print(f"  Data dir    : {DATA_DIR}")
    print("=" * 65 + "\n")

    station_data = {}

    for station_name, filename in STATIONS.items():
        csv_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(csv_path):
            print(f"  ✗ MISSING: {filename}  (skipping {station_name})")
            continue

        # If the file already has a "date" column and no "Year" column it was
        # processed in a previous run - load it directly and skip aggregation.
        probe = pd.read_csv(csv_path, nrows=1, dtype=str)
        if "date" in probe.columns and "Year" not in probe.columns:
            print(f"  Already processed: {station_name}")
            daily_df = pd.read_csv(csv_path)
            daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
            station_data[station_name] = daily_df
            n_days    = len(daily_df)
            n_missing = int(daily_df[["t2m","tmin","tmax","ws_10min"]].isna().any(axis=1).sum())
            print(f"    {n_days} days | {n_missing} days with any NaN")
            continue

        print(f"  Processing: {station_name}")

        hourly_df = load_hourly(csv_path)
        daily_df  = aggregate_daily(hourly_df)
        daily_df.to_csv(csv_path, index=False)

        n_days    = len(daily_df)
        n_missing = int(daily_df[["t2m","tmin","tmax","ws_10min"]].isna().any(axis=1).sum())
        print(f"    {n_days} days | {n_missing} days with any NaN | saved → {filename}")

        station_data[station_name] = daily_df

    # -----------------------------------------------------------------------
    # Missing-value summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Missing-value summary  (NaN count per column)")
    print("=" * 65)
    summary_df = build_missing_summary(station_data)
    pivot = summary_df.pivot(index="station", columns="column", values="missing_count")
    print(pivot.to_string())

    summary_path = os.path.join(DATA_DIR, "missing_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved → {summary_path}")
    print("\n✓ Phase 1 complete.")


if __name__ == "__main__":
    main()
