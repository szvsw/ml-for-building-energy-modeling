import requests
import boto3
import base64
import logging
import concurrent.futures
import json
import os
from typing import Literal
from uuid import UUID, uuid4

import geopandas as gpd
import pandas as pd
import requests
import numpy as np
import taichi as ti
from archetypal import UmiTemplateLibrary
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import Session
from botocore.config import Config as BotocoreConfig
from dotenv import get_key, load_dotenv
from fastapi import Depends, FastAPI, File, UploadFile, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ladybug.epw import EPW
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from weather.weather import extract

from umi.ubem import UBEM

load_dotenv()

app = FastAPI(docs_url="/api/docs", redoc_url="/api/redoc")
api = APIRouter(prefix="/api")

# TODO: move this to a config file
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_UBEM_ENDPOINT = os.getenv("RUNPOD_UBEM_ENDPOINT")

# to run the app, run the following command in the terminal:
# uvicorn api.main:app --reload --host 0.0.0.0 --port 8001


@api.get("/")
def read_root():
    return {"message": "healthy"}


class GISColumns(BaseModel):
    id_col: str = "OBJECTID"
    height_col: str = "HEIGHT"
    wwr_col: str = "wwr"
    template_name_col: str = "template_name"


class Library(BaseModel):
    templates: dict = {}
    schedules: list[list[list[float]]] = [[[]]]


# Function to sign a request
def sign_request(url, method, data, service, region):
    session = Session()
    credentials = session.get_credentials()
    creds = credentials.get_frozen_credentials()
    request = AWSRequest(method=method, url=url, data=data)
    SigV4Auth(creds, service, region).add_auth(request)
    return dict(request.headers)


def submit_eplus_shoeboxes(
    payload: dict,
    service: str,
    region: str,
    lambda_url: str,
    headers: dict = None,
):
    is_cloud_lambda = lambda_url.endswith(".aws/")
    if is_cloud_lambda:
        # Sign the request
        headers = sign_request(
            url=lambda_url,
            method="POST",
            data=payload,
            service=service,
            region=region,
        )
        payload["is_cloud_lambda"] = True

    else:
        payload["is_cloud_lambda"] = False
        payload = json.dumps({"body": payload})

    # Send the request
    req_kwargs = {
        "data": payload,
    }

    if headers is None:
        headers = {}
    if "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    req_kwargs["headers"] = headers
    if not is_cloud_lambda:
        response = requests.post(lambda_url, **req_kwargs)
        return response.json()
    else:
        config = BotocoreConfig(
            read_timeout=900, connect_timeout=900, retries={"max_attempts": 0}
        )
        session = boto3.Session()
        client = session.client("lambda", config=config, region_name="us-east-1")
        response = client.invoke(
            FunctionName="arn:aws:lambda:us-east-1:437256682840:function:eplus-shoebox-batch",
            InvocationType="RequestResponse",
            Payload=json.dumps({"body": payload}),
        )
        if "FunctionError" in response:
            logger.error(response["FunctionError"])
            logger.error(response["Payload"].read())
        data = json.loads(response["Payload"].read())
        return data


@api.post("/ubem")
def build_ubem(
    gis_file: UploadFile = File(...),
    epw_file: UploadFile = File(...),
    utl_file: UploadFile = File(...),
    gis_columns: GISColumns = Depends(),
    uuid: str = "",
    lib_mode: Literal["utl", "ml"] = "utl",
    simulator: Literal["eplus", "ml"] = "ml",
):
    ti.init(arch=ti.cpu)
    gdf = gpd.read_file(gis_file.file)
    tmp = f"data/backend/temp/{uuid}"
    os.makedirs(tmp, exist_ok=True)
    with open(f"{tmp}/epw.epw", "wb") as f:
        f.write(epw_file.file.read())
    epw = EPW(f"{tmp}/epw.epw")
    lib_data = utl_file.file.read()
    if lib_mode == "utl":
        utl = UmiTemplateLibrary.loads(lib_data, name="UBEMLib")
    elif lib_mode == "ml":
        lib = json.loads(lib_data)
        lib = Library(**lib)
        shoebox_features = pd.DataFrame.from_dict(lib.templates, orient="tight")
        template_schedules = np.array(lib.schedules, dtype=np.float32)
        utl = (shoebox_features, template_schedules)
    else:
        raise ValueError(f"lib_mode must be one of 'utl' or 'ml', not {lib_mode}")
    # TODO: UBEM should accept pydantic config inputs
    ubem = UBEM(
        gdf=gdf,
        epw=epw,
        **gis_columns.model_dump(),
        template_lib=utl,
        sensor_spacing=3,
        shoebox_width=3,
        floor_to_floor_height=4,
        perim_offset=4,
        shoebox_gen_type="edge_unshaded",
    )
    shoebox_features, schedules, climate = ubem.prepare_for_surrogate()
    if lib_mode == "utl":
        # overwrite these because they are not computed correctly
        shoebox_features["Infiltration"] = 0.00005
        shoebox_features["WindowShgc"] = 0.5
        shoebox_features["VentilationMode"] = 1
        logger.warning(
            "Overwriting Infiltration, WindowShgc, and VentilationMode since they are not computed correctly in UTL mode."
        )
    if simulator == "ml":
        features_data_dict = shoebox_features.to_dict(orient="tight")
        schedules_data_list = schedules.tolist()
        climate_data_list = climate.tolist()
        job_input = {
            "features": features_data_dict,
            "schedules": schedules_data_list,
            "climate": climate_data_list,
        }
        job = {
            "input": job_input,
        }

        endpoint_url = RUNPOD_UBEM_ENDPOINT
        sync = False
        run_url = f"{endpoint_url}/runsync" if sync else f"{endpoint_url}/run"

        response = requests.post(
            run_url, json=job, headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        )
        result = response.json()
    elif simulator == "eplus":
        # TODO: convert into background task and add building level aggregation
        # TODO: possibly move building level aggregation into a separate fn ala predict_ubem for ml

        if shoebox_features["core_depth"].min() < 1:
            logger.warning("core_depth must be >= 1, setting to 1!")
            shoebox_features["core_depth"] = shoebox_features["core_depth"].clip(1, 100)
        shoebox_features = pd.concat([shoebox_features] * 3, axis=0)
        building_ids = shoebox_features["building_id"].unique()
        building_dfs = [
            shoebox_features[shoebox_features["building_id"] == id]
            for id in building_ids
        ]
        # split each dataframe within building_dfs into chunks of 5 shoeboxes (or less)
        chunk_size = 4
        building_dfs = [
            df.iloc[i : i + chunk_size]
            for df in building_dfs
            for i in range(0, len(df), chunk_size)
        ]
        assert all(
            [len(df) < 50 for df in building_dfs]
        ), "too many shoeboxes per lambda worker, max is 50 due to timeout restrictions"
        building_templates_idxs = [df.template_idx.unique()[0] for df in building_dfs]
        # TODO: fuckkk this wil blow up mem quickly... possibly?
        schedules = [schedules[idx] for idx in building_templates_idxs]

        building_df_dicts = [df.to_dict(orient="tight") for df in building_dfs]
        schedules_lists = [schedule.tolist() for schedule in schedules]
        with open(f"{tmp}/epw.epw", "rb") as f:
            climate = base64.b64encode(f.read()).decode("utf-8")

        service = "lambda"
        region = os.getenv("AWS_REGION", "us-east-1")
        lambda_url = os.getenv("AWS_LAMBDA_SHOEBOX_URL")
        is_cloud_lambda = lambda_url.endswith(".aws/")

        n_buildings = len(building_ids)
        n_chunks = len(building_dfs) if is_cloud_lambda else 5
        # worker_count = 100 if is_cloud_lambda else 1
        worker_count = n_chunks if is_cloud_lambda else 1
        logger.info(
            f"Submitting {n_chunks} requests which represent {n_buildings} buildings using {worker_count} threads"
        )
        job_inputs = [
            {
                "climate": climate,
                "features": building_df_dict,
                "schedules": schedules_list,
            }
            for building_df_dict, schedules_list in zip(
                building_df_dicts[:n_chunks], schedules_lists[:n_chunks]
            )
        ]

        import time

        s = time.time()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:
            predicted = list(
                executor.map(
                    submit_eplus_shoeboxes,
                    job_inputs,
                    [service] * n_chunks,
                    [region] * n_chunks,
                    [lambda_url] * n_chunks,
                )
            )
            dfs = [
                pd.DataFrame.from_dict(df_data, orient="tight") for df_data in predicted
            ]
            df = pd.concat(dfs, axis=0)
            end = time.time()
            logger.info(
                f"took {(end-s):0.3f} seconds for {len(shoebox_features)} shoeboxes"
            )

            return {"message": "success"}
    else:
        raise ValueError(f"simulator must be one of 'eplus' or 'ml', not {simulator}")

    for file in os.listdir(tmp):
        os.remove(f"{tmp}/{file}")
    return result


@api.get("/ubem/status/{job_id}")
def check_status(job_id: str):
    endpoint_url = RUNPOD_UBEM_ENDPOINT
    status_url = f"{endpoint_url}/status/{job_id}"
    response = requests.get(
        status_url, headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    )
    data = response.json()

    return data


@api.post("/building")
def build_building(
    epw_file: UploadFile = File(...),
    template_file: UploadFile = File(...),
    uuid: str = "",
):
    # TODO: most of this logic should be its own class inside of umi.ubem
    tmp = f"data/backend/temp/{uuid}"
    os.makedirs(tmp, exist_ok=True)
    with open(f"{tmp}/epw.epw", "wb") as f:
        f.write(epw_file.file.read())
    epw = EPW(f"{tmp}/epw.epw")
    template_data = json.load(template_file.file)
    schedules = np.array(template_data["schedules"])
    schedules = schedules.reshape(-1, *schedules.shape)
    climate = extract(epw)

    features = template_data["features"]
    features["template_idx"] = 0
    features["template_name"] = "test"
    features["building_id"] = 0
    for i in range(12):
        features[f"shading_{i}"] = 0
    floor_to_floor_height = features["height"]
    building_width = features["building_width"]
    building_length = features["building_length"]
    building_perim_length = 2 * (building_width + building_length)
    facade_area = building_perim_length * floor_to_floor_height
    perim_offset = 4
    core_width = building_width - 2 * perim_offset
    core_length = building_length - 2 * perim_offset
    has_core = core_width > 0 and core_length > 0
    footprint_area = building_width * building_length
    core_area = 0 if not has_core else core_width * core_length
    perim_area = facade_area - core_area
    perim_area_to_facade_area = perim_area / facade_area
    core_area_to_perim_area = core_area / perim_area
    sb_width = 3
    features["width"] = sb_width
    sb_facade_area = sb_width * floor_to_floor_height
    sb_perim_area = perim_area_to_facade_area * sb_facade_area
    sb_perim_depth = sb_perim_area / sb_width
    sb_core_depth = 0 if not has_core else core_area_to_perim_area * sb_perim_depth
    features["perim_depth"] = sb_perim_depth
    features["core_depth"] = sb_core_depth

    south_weight = building_width / building_perim_length
    east_weight = building_length / building_perim_length
    north_weight = south_weight
    west_weight = east_weight
    edge_weights = {
        "S": south_weight,
        "E": east_weight,
        "N": north_weight,
        "W": west_weight,
    }

    n_floors = features["n_floors"]

    shoeboxes = []
    for i, orientation in enumerate(edge_weights.keys()):
        edge_weight = edge_weights[orientation]
        orientation_data = features.copy()
        orientation_data["edge_weight"] = edge_weight
        orientation_data["orientation"] = i * np.pi / 2
        orientation_data["wwr"] = features[f"wwr_{orientation}"]
        if n_floors == 1:
            sb_data = orientation_data.copy()
            sb_data["floor_weight"] = 1
            sb_data["roof_2_footprint"] = 1
            sb_data["ground_2_footprint"] = 1
            shoeboxes.append(sb_data)
        elif n_floors == 2:
            bottom_data = orientation_data.copy()
            top_data = orientation_data.copy()
            bottom_data["floor_weight"] = 0.5
            top_data["floor_weight"] = 0.5
            bottom_data["roof_2_footprint"] = 0
            bottom_data["ground_2_footprint"] = 1
            top_data["roof_2_footprint"] = 1
            top_data["ground_2_footprint"] = 0
            shoeboxes.append(bottom_data)
            shoeboxes.append(top_data)
        elif n_floors > 2:
            bottom_data = orientation_data.copy()
            middle_data = orientation_data.copy()
            top_data = orientation_data.copy()
            bottom_data["floor_weight"] = 1 / n_floors
            middle_data["floor_weight"] = (n_floors - 2) / n_floors
            top_data["floor_weight"] = 1 / n_floors
            bottom_data["roof_2_footprint"] = 0
            bottom_data["ground_2_footprint"] = 1
            middle_data["roof_2_footprint"] = 0
            middle_data["ground_2_footprint"] = 0
            top_data["roof_2_footprint"] = 1
            top_data["ground_2_footprint"] = 0
            shoeboxes.append(bottom_data)
            shoeboxes.append(middle_data)
            shoeboxes.append(top_data)

    features = pd.DataFrame(data=shoeboxes)
    features["weight"] = features["floor_weight"] * features["edge_weight"]
    features["core_weight"] = features["core_depth"] / (
        features["core_depth"] + features["perim_depth"]
    )
    features["perim_weight"] = features["perim_depth"] / (
        features["core_depth"] + features["perim_depth"]
    )
    features_data_dict = features.to_dict(orient="tight")
    schedules_data_list = schedules.tolist()
    climate_data_list = climate.tolist()
    job_input = {
        "features": features_data_dict,
        "schedules": schedules_data_list,
        "climate": climate_data_list,
    }
    job = {
        "input": job_input,
    }

    endpoint_url = RUNPOD_UBEM_ENDPOINT
    sync = True
    run_url = f"{endpoint_url}/runsync" if sync else f"{endpoint_url}/run"

    response = requests.post(
        run_url, json=job, headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    )
    response_data = response.json()
    if "error" in response_data:
        raise HTTPException(status_code=400, detail=response_data["error"])
    output = response_data["output"]
    return output


@api.post("/eplus-shoeboxes")
def eplus_shoeboxes():
    payload = {"key": "value"}
    service = "lambda"
    region = os.getenv("AWS_REGION", "us-east-1")
    lambda_url = os.getenv("AWS_LAMBDA_SHOEBOX_URL")
    is_cloud_lambda = lambda_url.endswith(".aws/")

    n = 10
    worker_count = 10 if is_cloud_lambda else 1
    import time

    s = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        predicted = list(
            executor.map(
                submit_eplus_shoeboxes,
                [payload] * n,
                [service] * n,
                [region] * n,
                [lambda_url] * n,
            )
        )
    e = time.time()
    print(f"took {(e-s):0.3f} seconds, average {(e-s)/n:0.3f} seconds per job")
    return {"message": "success"}


app.include_router(api)
