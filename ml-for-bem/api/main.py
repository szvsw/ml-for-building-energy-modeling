from fastapi import FastAPI, File, UploadFile, Depends
import os
from fastapi.responses import JSONResponse
from dotenv import load_dotenv, get_key
from uuid import uuid4, UUID

import requests
import taichi as ti
import geopandas as gpd
import pandas as pd
from ladybug.epw import EPW
from archetypal import UmiTemplateLibrary
from umi.ubem import UBEM
from pydantic import BaseModel

app = FastAPI()
# TODO: move this to a config file
RUNPOD_API_KEY = get_key(".env", key_to_get="RUNPOD_API_KEY")
RUNPOD_UBEM_ENDPOINT = get_key(".env", key_to_get="RUNPOD_UBEM_ENDPOINT")

# to run the app, run the following command in the terminal:
# uvicorn api.main:app --reload --host 0.0.0.0 --port 8001


@app.get("/")
def read_root():
    return {"message": "Hello World"}


class GISColumns(BaseModel):
    id_col: str = "OBJECTID"
    height_col: str = "HEIGHT"
    wwr_col: str = "wwr"
    template_name_col: str = "template_name"


@app.post("/ubem")
def build_ubem(
    gis_file: UploadFile = File(...),
    epw_file: UploadFile = File(...),
    utl_file: UploadFile = File(...),
    gis_columns: GISColumns = Depends(),
    uuid: str = str,
):
    ti.init(arch=ti.cpu)
    gdf = gpd.read_file(gis_file.file)
    tmp = f"data/backend/temp/{uuid}"
    os.makedirs(tmp, exist_ok=True)
    with open(f"{tmp}/epw.epw", "wb") as f:
        f.write(epw_file.file.read())
    epw = EPW(f"{tmp}/epw.epw")
    utl = UmiTemplateLibrary.loads(utl_file.file.read(), name="UBEMLib")
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
    features, schedules, climate = ubem.prepare_for_surrogate()
    features["Infiltration"] = 0.00005
    features["WindowShgc"] = 0.5
    features["VentilationMode"] = 1
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
    sync = False
    run_url = f"{endpoint_url}/runsync" if sync else f"{endpoint_url}/run"

    response = requests.post(
        run_url, json=job, headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    )

    os.removedirs(tmp)
    return response.json()


@app.get("/ubem/status/{job_id}")
def check_status(job_id: str):
    endpoint_url = RUNPOD_UBEM_ENDPOINT
    status_url = f"{endpoint_url}/status/{job_id}"
    response = requests.get(
        status_url, headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    )
    data = response.json()

    return data
