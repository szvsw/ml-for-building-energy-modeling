import pandas as pd
from ladybug.epw import EPW
from typing import List
from utils.constants import EPW_MAP_PATH, EPW_TESTING_LIST_PATH, EPW_TRAINING_LIST_PATH
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
    epw,
    timeseries_names: List[str] = [
        "dry_bulb_temperature",
        "dew_point_temperature",
        "relative_humidity",
        "wind_direction",
        "wind_speed",
        "direct_normal_radiation",
        "diffuse_horizontal_radiation",
    ],
):
    channels = []
    for channel_name in timeseries_names:
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
    experiment_name = "weather/test"
    file_name = "global_climate_array_test"
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
    ]
    epw_folder_name = "global_epws_indexed"
    epw_root = Path(__file__).parent.parent / "data" / "epws"
    epw_dir = epw_root / epw_folder_name
    save_path = epw_root / file_name

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
