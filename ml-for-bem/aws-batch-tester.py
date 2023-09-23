import os
from datetime import datetime
import h5py
import json

import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

from storage import upload_to_bucket

import logging

logging.basicConfig()
logger = logging.getLogger("Sampler Sim Test")


from schema import (
    Schema,
    TimeSeriesOutput,
    ShoeboxGeometryParameter,
    ShoeboxOrientationParameter,
    BuildingTemplateParameter,
    WhiteboxSimulation,
    WindowParameter,
    SchedulesParameters,
)
from nrel_uitls import (
    ResStockConfiguration,
    CLIMATEZONES,
    CLIMATEZONES_LIST,
    WINDTYPES,
    RESTYPES,
)

with open("./data/city_map.json", "r") as f:
    city_map = json.load(f)

timeseries = [
    TimeSeriesOutput(
        name="DistrictCooling",
        key_name="Cooling:DistrictCooling",
        key="OUTPUT:METER",
        freq="Monthly",
        store_output=True,
    ),
    TimeSeriesOutput(
        name="DistrictHeating",
        key_name="Heating:DistrictHeating",
        key="OUTPUT:METER",
        freq="Monthly",
        store_output=True,
    ),
    TimeSeriesOutput(
        name="Supply Air Heating",
        var_name="Zone Ideal Loads Supply Air Total Heating Energy",
        key="OUTPUT:VARIABLE",
        freq="Hourly",
        store_output=True,
    ),
    TimeSeriesOutput(
        name="Supply Air Cooling",
        var_name="Zone Ideal Loads Supply Air Total Cooling Energy",
        key="OUTPUT:VARIABLE",
        freq="Hourly",
        store_output=True,
    ),
    TimeSeriesOutput(
        name="OA Heating",
        var_name="Zone Ideal Loads Outdoor Air Total Heating Energy",
        key="OUTPUT:VARIABLE",
        freq="Hourly",
        store_output=True,
    ),
    TimeSeriesOutput(
        name="OA Cooling",
        var_name="Zone Ideal Loads Outdoor Air Total Cooling Energy",
        key="OUTPUT:VARIABLE",
        freq="Hourly",
        store_output=True,
    ),
    # Zone Ideal Loads Supply Air Total Heating Energy
    # Zone Ideal Loads Zone Total Heating Energy
]
schema = Schema(timeseries_outputs=timeseries)

s_id = int(os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", -1))

if s_id == -1:
    s_id = 1

for i in range((20 * s_id) if s_id != 0 else 1):
    logger.info(f"---Starting simulation {i}---")
    storage_vector = schema.generate_empty_storage_vector()
    # just using
    # TODO: orientation
    # TODO: setpoint value overlaps
    # TODO: Fix template Selector
    # TODO: window u value coming out different
    schema.update_storage_vector(
        storage_vector, parameter="climate_zone", value=CLIMATEZONES["2A"]
    )
    schema.update_storage_vector(storage_vector, parameter="vintage", value=1920)
    schema.update_storage_vector(
        storage_vector,
        parameter="program_type",
        value=RESTYPES["Multi-Family with 5+ Units"],
    )
    schema.update_storage_vector(
        storage_vector,
        parameter="base_epw",
        value=city_map["CA, Los Angeles"]["idx"],
    )
    schema.update_storage_vector(storage_vector, "height", 3)
    schema.update_storage_vector(storage_vector, "width", 3)
    schema.update_storage_vector(storage_vector, "facade_2_footprint", 0.3)
    schema.update_storage_vector(storage_vector, "perim_2_footprint", 0.5)
    schema.update_storage_vector(storage_vector, "roof_2_footprint", 0.05)
    schema.update_storage_vector(storage_vector, "footprint_2_ground", 0.5)
    schema.update_storage_vector(storage_vector, "wwr", 0.3)
    schema.update_storage_vector(storage_vector, "orientation", 0)
    schema.update_storage_vector(storage_vector, "Infiltration", 1.0)
    schema.update_storage_vector(storage_vector, "HeatingSetpoint", 18)
    schema.update_storage_vector(storage_vector, "CoolingSetpoint", 24)
    schema.update_storage_vector(storage_vector, "PeopleDensity", 0.05)
    schema.update_storage_vector(storage_vector, "LightingPowerDensity", 3)
    schema.update_storage_vector(storage_vector, "EquipmentPowerDensity", 7)
    schema.update_storage_vector(storage_vector, "RoofRValue", 2)
    schema.update_storage_vector(storage_vector, "SlabRValue", 2)
    schema.update_storage_vector(storage_vector, "FacadeRValue", 2)
    schema.update_storage_vector(storage_vector, "FacadeMass", 10000)
    schema.update_storage_vector(storage_vector, "RoofMass", 10000)

    # NEW VALUES
    schema.update_storage_vector(storage_vector, "WindowSettings", 2)
    schema.update_storage_vector(storage_vector, "shading_seed", 2)
    schema.update_storage_vector(storage_vector, "EconomizerSettings", 1)
    schema.update_storage_vector(storage_vector, "RecoverySettings", 2)

    schedules = schema["schedules"].extract_storage_values(storage_vector)
    sched_demo = "BASELINE"
    if sched_demo == "BASELINE":
        # Equipment
        # pass
        # Occupancy
        schedules[1, SchedulesParameters.op_indices["noise"]] = 0.2
        # Lights
        schedules[2, SchedulesParameters.op_indices["invert"]] = 1

    whitebox_sim = WhiteboxSimulation(schema, storage_vector)
    res_hourly, res_monthly = whitebox_sim.simulate()

    res_hourly.to_hdf("test.hdf5", key=f"hourly_{i:05d}", mode="w" if i == 0 else "a")
    res_monthly.to_hdf("test.hdf5", key=f"monthly_{i:05d}", mode="a")

    with h5py.File("test.hdf5", mode="r+") as f:
        f.create_dataset(
            f"storage_vector_{i:05d}",
            shape=storage_vector.shape,
            data=storage_vector,
        )

if s_id != -1:
    upload_to_bucket(
        blob_name=f"batch-sims/array-test/multi/{(10*s_id):05d}.hdf5",
        file_name="test.hdf5",
    )
else:
    upload_to_bucket(
        blob_name=f"batch-sims/array-test/no_id/timestamp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.hdf5",
        file_name="test.hdf5",
    )
