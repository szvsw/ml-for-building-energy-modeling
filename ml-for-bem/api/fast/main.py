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
from dotenv import get_key, load_dotenv
from fastapi import Depends, FastAPI, File, UploadFile, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ladybug.epw import EPW
from pydantic import BaseModel

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


@api.post("/ubem")
def build_ubem(
    gis_file: UploadFile = File(...),
    epw_file: UploadFile = File(...),
    utl_file: UploadFile = File(...),
    gis_columns: GISColumns = Depends(),
    uuid: str = "",
    lib_mode: Literal["utl", "ml"] = "utl",
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
        utl = UmiTemplateLibrary.loads(utl_file.file.read(), name="UBEMLib")
    elif lib_mode == "ml":
        lib = json.loads(lib_data)
        lib = Library(**lib)
        template_features = pd.DataFrame.from_dict(lib.templates, orient="tight")
        template_schedules = np.array(lib.schedules, dtype=np.float32)
        utl = (template_features, template_schedules)
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
    template_features, schedules, climate = ubem.prepare_for_surrogate()
    if lib_mode == "utl":
        # overwrite these because they are not computed correctly
        template_features["Infiltration"] = 0.00005
        template_features["WindowShgc"] = 0.5
        template_features["VentilationMode"] = 1
    features_data_dict = template_features.to_dict(orient="tight")
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

    for file in os.listdir(tmp):
        os.remove(f"{tmp}/{file}")
    return response.json()


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


app.include_router(api)
