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
from fastapi import Depends, FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from ladybug.epw import EPW
from pydantic import BaseModel

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


app.include_router(api)
