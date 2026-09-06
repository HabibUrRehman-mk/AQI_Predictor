"""Backfill historical Open-Meteo data into the Hopsworks feature group.

Usage:
    python src/workflows/historical_backfill.py \
        --start-date 2024-01-01 \
        --end-date 2024-12-31

The requested range is expanded by 72 hours on both sides so lag, rolling,
and forecast-target features are complete at the requested boundaries.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta

import hopsworks
import pandas as pd
import requests

from data_to_feature_store import (
    CITY_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    HOPSWORKS_HOST,
    PROJECT_NAME,
    engineer_features,
)

LATITUDE = 31.4187
LONGITUDE = 73.0791
CHUNK_DAYS = 14
CONTEXT_HOURS = 72
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise argparse.ArgumentTypeError("Date must use YYYY-MM-DD format") from error


def date_chunks(start_date: date, end_date: date):
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), end_date)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def fetch_chunk(start_date: date, end_date: date) -> pd.DataFrame:
    common = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "auto",
    }
    weather_params = {
        **common,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
    }
    air_params = {
        **common,
        "hourly": "pm10,pm2_5,nitrogen_dioxide,ozone,us_aqi",
    }

    weather_response = requests.get(WEATHER_ARCHIVE_URL, params=weather_params, timeout=60)
    weather_response.raise_for_status()
    air_response = requests.get(AIR_QUALITY_URL, params=air_params, timeout=60)
    air_response.raise_for_status()

    weather = pd.DataFrame(weather_response.json()["hourly"])
    air = pd.DataFrame(air_response.json()["hourly"])
    data = pd.merge(weather, air, on="time", how="inner")
    data["time"] = pd.to_datetime(data["time"])
    data["city"] = CITY_NAME
    return data


def fetch_range(start_date: date, end_date: date) -> pd.DataFrame:
    chunks = []
    for chunk_start, chunk_end in date_chunks(start_date, end_date):
        log.info("Fetching %s to %s", chunk_start, chunk_end)
        chunks.append(fetch_chunk(chunk_start, chunk_end))

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["city", "time"])


def backfill(start_date: date, end_date: date, dry_run: bool = False) -> int:
    if end_date < start_date:
        raise ValueError("end-date must be on or after start-date")

    expanded_start = start_date - timedelta(hours=CONTEXT_HOURS)
    # API requests are date-granular, so include the complete end date plus
    # enough full days for the final 72-hour target window.
    expanded_end = end_date + timedelta(days=(CONTEXT_HOURS // 24) + 1)
    raw_data = fetch_range(expanded_start, expanded_end)
    if raw_data.empty:
        raise RuntimeError("The data provider returned no rows for the requested range")

    features = engineer_features(raw_data)
    features["time"] = pd.to_datetime(features["time"])
    requested = features[features["time"].between(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date) + timedelta(days=1) - timedelta(seconds=1),
    )].copy()
    requested = requested.dropna(subset=["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"])

    log.info("Prepared %d complete feature rows for %s to %s", len(requested), start_date, end_date)
    if dry_run:
        log.info("Dry run enabled; no Hopsworks write performed")
        return len(requested)

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is required unless --dry-run is used")

    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=api_key,
        project=os.getenv("HOPSWORKS_PROJECT", PROJECT_NAME),
    )
    feature_store = project.get_feature_store()
    feature_group = feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )
    feature_group.insert(requested, write_options={"wait_for_job": True})
    log.info("Backfill inserted successfully")
    return len(requested)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill historical AQI features into Hopsworks")
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date)
    parser.add_argument("--dry-run", action="store_true", help="Fetch and engineer data without writing to Hopsworks")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        backfill(args.start_date, args.end_date, dry_run=args.dry_run)
    except Exception:
        log.exception("Historical backfill failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
