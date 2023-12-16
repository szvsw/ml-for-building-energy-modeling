import pandas as pd
from ladybug.epw import EPW
from pvlib.solarposition import get_solarposition
from datetime import datetime
import pytz
from typing import List
from utils.constants import EPW_MAP_PATH, EPW_TESTING_LIST_PATH, EPW_TRAINING_LIST_PATH
from timezonefinder import TimezoneFinder
from tqdm.auto import tqdm
import numpy as np
import requests
import time


# TODO: hook this into better pathing/aws
def make_population_sorted_csv():
    # Get the list of all cities

    df = pd.read_csv(
        Path(__file__).parent.parent / "data" / "worldcities.csv"
    ).sort_values(by="population", ascending=False)

    # Pick The 8000 most populous
    df = df[:8000]

    # Add a column for CZ
    df["CZ"] = ""

    # Get the CZ from ubem.io
    for ix, row in tqdm(df.iterrows()):
        print(f"Checking CZ for ", row["city"])
        lat, lng = row["lat"], row["lng"]
        try:
            res = requests.get(
                f"https://app.ubem.io/api/umibuilder/get_climate_zone?Lat={lat}&Lon={lng}"
            )
            df.loc[ix, "CZ"] = list(res.json().values())[0]["CZ"]
        except:
            print("ERROR", row["city"])
            pass
        # Sleep to prevent accidental DDoS
        time.sleep(0.1)

    # Save DF
    df.to_csv(Path(__file__).parent.parent / "data" / "8k_cities_with_CZ.csv")


# TODO: hook this into better pathing/aws
def download_EPWS_from_source():
    import random
    import pandas as pd
    from pathlib import Path
    from pyumi.epw import Epw
    import time

    # Get list of 8k most populous cities with their CZs
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "8k_cities_with_CZ.csv")

    # Drop any which errored when fetching cz
    df = df[~(df["CZ"].isna())]

    # Sort by CZ, then population
    df = df.sort_values(by=["CZ", "population"], ascending=False)

    # Make a dict of columns and add a few then create new df
    cols = {col: [] for col in df.columns}
    cols["idx"] = []
    cols["slug"] = []
    top_cities = pd.DataFrame(cols)

    # Get the nth most populous city in each climate zone, then fetch epw
    for i in range(50):
        for cz in df["CZ"].unique():
            # Get the cities in each climate zone
            czdf = df[df["CZ"] == cz]

            # make sure the nth most populous city is defined for this cz
            if i < len(czdf):
                # Get the lat lon and try to download epw
                city = czdf.iloc[i]
                lat = city["lat"]
                lng = city["lng"]
                try:
                    time.sleep(random.random() * 5 + 1)
                    city_epw = Epw.from_nrel(lat=lat, lon=lng)
                except BaseException as e:
                    print(e)
                    print("FAILED:", city["city"])
                    continue
                else:
                    # Save the epw and add it to the array
                    print("SUCCESS", city["city"])
                    slug = f"cityidx_{len(top_cities):04d}_{city_epw.name}"
                    out_path = (
                        Path(__file__).parent.parent
                        / "data"
                        / "global_epws_indexed"
                        / slug
                    )
                    city_epw.save(out_path)
                    city["slug"] = slug
                    city["idx"] = len(top_cities)
                    top_cities.loc[len(top_cities)] = city

    # save the results and paths
    top_cities.to_csv(
        Path(__file__).parent.parent / "data" / "epws" / "global_epws_indexed.csv"
    )


def extract(
    epw: EPW,
    timeseries_names: List[str] = [
        "dry_bulb_temperature",
        "dew_point_temperature",
        "relative_humidity",
        "wind_direction",
        "wind_speed",
        "direct_normal_radiation",
        "diffuse_horizontal_radiation",
        "solar_azimuth",
        "solar_elevation",
        "latitude",
        "longitude",
    ],
):
    channels = []
    latitude = epw.location.latitude
    longitude = epw.location.longitude
    if "solar_azimuth" in timeseries_names:
        assert (
            "solar_elevation" in timeseries_names
        ), "Must include elevation with azimuth"
    if "solar_elevation" in timeseries_names:
        assert (
            "solar_azimuth" in timeseries_names
        ), "Must include azimuth with elevation"
    if "solar_azimuth" in timeseries_names or "solar_elevation" in timeseries_names:
        # create datetime index for the whole year in 1 hr increments, resulting in 8760 points
        dt_index = pd.date_range(start="1/1/2019", end="1/1/2020", freq="1H")[:-1]

        # Use TimezoneFinder to get the timezone string
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(
            lat=latitude, lng=longitude
        )  # This returns something like 'America/New_York'

        # timezone
        tz = pytz.timezone(timezone_str)

        # Get the datetime for New Year's Day in a specific year
        new_years_day = datetime(year=2019, month=1, day=1)

        # Localize the New Year's Day datetime to the timezone
        localized_new_years_day = tz.localize(new_years_day, is_dst=None)

        # Get the UTC offset
        utc_offset = localized_new_years_day.utcoffset()
        # get the utc offset hours
        utc_offset_hours = utc_offset.total_seconds() // 3600

        # Localize your DateTimeIndex to the found timezone
        # localized_datetime_index = dt_index.tz_localize(
        #     timezone_str,
        #     ambiguous="infer",  # Standard time for ambiguous (fall back)
        #     nonexistent="shift_forward",  # Shift forward for nonexistent (spring forward)
        # )

        # solar_position = get_solarposition(
        #     localized_datetime_index,
        #     latitude,
        #     longitude,
        # )
        solar_position_utc = get_solarposition(
            dt_index,
            latitude,
            longitude,
        )

        azimuths = np.array(solar_position_utc.azimuth.values)
        elevations = np.array(solar_position_utc.elevation.values)

        # roll the azimuths and elevations accoring to the utc offset
        azimuths = np.roll(azimuths, int(utc_offset_hours))
        elevations = np.roll(elevations, int(utc_offset_hours))
        azimuths_cos = np.cos(np.radians(azimuths))
        azimuths_sin = np.sin(np.radians(azimuths))
        elevations_cos = np.cos(np.radians(elevations))
        elevations_sin = np.sin(np.radians(elevations))

    for channel_name in timeseries_names:
        if channel_name == "solar_azimuth":
            channels.append(np.array(azimuths))
        elif channel_name == "solar_azimuth_cos":
            channels.append(azimuths_cos)
        elif channel_name == "solar_azimuth_sin":
            channels.append(azimuths_sin)
        elif channel_name == "solar_elevation_cos":
            channels.append(elevations_cos)
        elif channel_name == "solar_elevation_sin":
            channels.append(elevations_sin)
        elif channel_name == "solar_elevation":
            channels.append(np.array(elevations))
        elif channel_name == "latitude":
            channels.append(np.array([latitude] * 8760))
        elif channel_name == "longitude":
            channels.append(np.array([longitude] * 8760))
        elif channel_name == "latitude_cos":
            channels.append(np.cos(np.radians(np.array([latitude] * 8760))))
        elif channel_name == "latitude_sin":
            channels.append(np.sin(np.radians(np.array([latitude] * 8760))))
        elif channel_name == "longitude_cos":
            channels.append(np.cos(np.radians(np.array([longitude] * 8760))))
        elif channel_name == "longitude_sin":
            channels.append(np.sin(np.radians(np.array([longitude] * 8760))))
        else:
            channel = np.array(getattr(epw, channel_name).values)
            channels.append(channel)

    channels = np.vstack(channels)
    assert np.isnan(channels).sum() == 0
    return channels


def make_climate_array(
    epw_map_path: str, epw_dir: str, save_path: str, timeseries_names: List[str]
):
    epw_map = pd.read_csv(epw_map_path, index_col=0)
    weather_data = []
    for idx, row in tqdm(epw_map.iterrows()):
        assert row.slug == epw_map.iloc[idx].slug
        city_epw = EPW(f"{epw_dir}/{row.slug}")
        channels = extract(city_epw, timeseries_names)
        weather_data.append(channels)
    weather_data = np.array(weather_data)
    np.save(save_path, weather_data.astype(np.float32))


if __name__ == "__main__":
    import json
    from pathlib import Path
    import boto3
    import shutil

    s3 = boto3.client("s3")

    bucket = "ml-for-bem"
    experiment_name = "weather/v2"
    file_name = "global_climate_array"
    bucket_destination = f"{experiment_name}/{file_name}.npy"

    """
    Configuration
    """
    timeseries = [
        "dry_bulb_temperature",
        "dew_point_temperature",
        "relative_humidity",
        "wind_direction",
        "wind_speed",
        "direct_normal_radiation",
        "diffuse_horizontal_radiation",
        "solar_azimuth",
        "solar_elevation",
        "solar_azimuth_cos",
        "solar_azimuth_sin",
        "solar_elevation_cos",
        "solar_elevation_sin",
        "latitude_cos",
        "latitude_sin",
        "longitude_cos",
        "longitude_sin",
        "latitude",
        "longitude",
    ]
    epw_folder_name = "global_epws_indexed"
    epw_root = Path("data") / "epws"
    epw_dir = epw_root / epw_folder_name
    save_path = epw_root / f"local_{file_name}"

    # make the array and save it
    make_climate_array(
        EPW_MAP_PATH,
        epw_dir=epw_dir,
        save_path=save_path,
        timeseries_names=timeseries,
    )

    # upload to s3
    s3.upload_file(
        f"{str(save_path)}.npy",
        bucket,
        bucket_destination,
    )

    # upload the epw map as well
    s3.upload_file(
        EPW_MAP_PATH,
        bucket,
        f"{experiment_name}/epw_map.csv",
    )

    s3.upload_file(
        EPW_TRAINING_LIST_PATH,
        bucket,
        f"{experiment_name}/epw_training_list.csv",
    )
    s3.upload_file(
        EPW_TESTING_LIST_PATH,
        bucket,
        f"{experiment_name}/epw_testing_list.csv",
    )

    # convert timeseries to json and upload it as well
    timeseries_json = json.dumps(timeseries)
    s3.put_object(
        Body=timeseries_json,
        Bucket=bucket,
        Key=f"{experiment_name}/timeseries.json",
    )

    # zip the epws and upload them as well
    shutil.make_archive(epw_folder_name, "zip", epw_dir)
    s3.upload_file(
        f"{epw_folder_name}.zip",
        bucket,
        f"{experiment_name}/{epw_folder_name}.zip",
    )
