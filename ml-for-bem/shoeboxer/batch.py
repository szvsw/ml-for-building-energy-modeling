import os
from typing import Literal, List, Union
import click
import shutil
import pandas as pd
from uuid import uuid4
from utils.nrel_uitls import CLIMATEZONES
from utils.constants import (
    JOULES_TO_KWH,
    EPW_RELATIVE_PATH,
)
import json
from pathlib import Path
import numpy as np
from ladybug.epw import EPW
from shoeboxer.shoebox_config import ShoeboxConfiguration
from shoeboxer.builder import ShoeBox, template_dict
from shoeboxer.schedules import schedules_from_seed, default_schedules
from archetypal import parallel_process
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

data_root = Path("data")


def simulate(
    features: pd.Series,
    timeseries: np.ndarray,
    climate: Union[Path, str],
):
    parameter_dict = features.to_dict()
    hsp = parameter_dict["HeatingSetpoint"]
    csp = parameter_dict["CoolingSetpoint"]
    if hsp > csp:
        parameter_dict["HeatingSetpoint"] = csp
        parameter_dict["CoolingSetpoint"] = hsp

    shading_columns = sorted(
        [col for col in features.index if "shading" in col.lower()]
    )
    shading_vect = np.array([features[col] for col in shading_columns])

    """
    Make the Shoebox Config
    """
    shoebox_config = ShoeboxConfiguration()

    """
    Build the Shoebox
    """
    shoebox_config.adiabatic_partition_flag = 0
    shoebox_config.shading_vect = shading_vect
    shoebox_config.width = parameter_dict["width"]
    shoebox_config.height = parameter_dict["height"]
    shoebox_config.core_depth = parameter_dict["core_depth"]
    shoebox_config.perim_depth = parameter_dict["perim_depth"]
    shoebox_config.roof_2_footprint = parameter_dict["roof_2_footprint"]
    shoebox_config.ground_2_footprint = parameter_dict["ground_2_footprint"]
    shoebox_config.wwr = parameter_dict["wwr"]
    shoebox_config.orientation = parameter_dict["orientation"]

    # make dict for template
    template_datadict = template_dict(
        schedules=timeseries,
        PeopleDensity=parameter_dict["PeopleDensity"],
        LightingPowerDensity=parameter_dict["LightingPowerDensity"],
        EquipmentPowerDensity=parameter_dict["EquipmentPowerDensity"],
        Infiltration=parameter_dict["Infiltration"],
        VentilationPerArea=parameter_dict["VentilationPerArea"],
        VentilationPerPerson=parameter_dict["VentilationPerPerson"],
        VentilationMode=parameter_dict["VentilationMode"],
        HeatingSetpoint=parameter_dict["HeatingSetpoint"],
        CoolingSetpoint=parameter_dict["CoolingSetpoint"],
        RecoverySettings=parameter_dict["RecoverySettings"],
        EconomizerSettings=parameter_dict["EconomizerSettings"],
        FacadeRValue=parameter_dict["FacadeRValue"],
        FacadeMass=parameter_dict["FacadeMass"],
        RoofRValue=parameter_dict["RoofRValue"],
        RoofMass=parameter_dict["RoofMass"],
        SlabRValue=parameter_dict["SlabRValue"],
        WindowShgc=parameter_dict["WindowShgc"],
        WindowUValue=parameter_dict["WindowUValue"],
    )

    """
    Setup Simulation
    """
    sb_name = str(uuid4())
    output_dir = data_root / "sim_results" / sb_name
    os.makedirs(output_dir, exist_ok=True)

    sb = ShoeBox(
        name=sb_name,
        shoebox_config=shoebox_config,
        template_dict=template_datadict,
        epw=climate,
        output_directory=output_dir,
        change_summary=False,
    )

    idf = sb.idf(run_simulation=False)

    """
    Run Simulation
    """
    hourly_df, monthly_df = sb.simulate(idf, postprocess=True)
    monthly_df = sb.postprocess(monthly_df)

    shutil.rmtree(output_dir)
    return sb_name, monthly_df


def batch_sim(
    features: pd.DataFrame,
    timeseries: np.ndarray,
    climate: Union[Path, str],
):
    """
    Run a batch simulation which consumes the dataframe

    Args:
        features (pd.DataFrame): dataframe of features which are tabular (e.g. geometry and hsp and csp)
        timeseries (np.ndarray): array of schedules data (e.g. people, lighting, equipment, etc.) for a single archetype (3,8760)
        climate (Union[Path, str]): path to epw file or string of climate zone name

    Returns:
        results (pd.DataFrame): dataframe of results
    """

    # make a dataframe to store results
    results = pd.DataFrame()

    # iterate over the dataframe
    for index, row in features.iterrows():
        id, monthly_results = simulate(
            features=row,
            timeseries=timeseries,
            climate=climate,
        )

        if len(results) == 0:
            # set the result
            results = pd.DataFrame(monthly_results)
            results = results.T
            # set the index to be a multi index with column names from keys of simple_dict and values from values of simple_dict
            index = (id, *(v for v in row.values))
            results.index = pd.MultiIndex.from_tuples(
                [index],
                names=["id"] + row.index.values.tolist(),
            )
        else:
            # make the multi-index of features
            index = (id, *(v for v in row.values))
            # set the result
            results.loc[index] = monthly_results

    return results


if __name__ == "__main__":
    from pathlib import Path

    from archetypal import settings

    # Check if we are running on Windows or Linux using os
    if os.name == "nt":
        settings.ep_version == "22.2.0"
        settings.energyplus_location = Path("C:/EnergyPlusV22-2-0")
    else:
        settings.ep_version == "22.2.0"
        settings.energyplus_location = Path("/usr/local/EnergyPlus-22-2-0")

    with open(f"app/template-defaults.json", "r") as f:
        template_defaults = json.load(f)
    df = pd.DataFrame([template_defaults for _ in range(5)])
    df["wwr"] = 0.4
    df["width"] = 3
    df["perim_depth"] = 5
    df["core_depth"] = 4
    df["orientation"] = 0
    df["roof_2_footprint"] = 1
    df["ground_2_footprint"] = 1
    for i in range(24):
        df[f"shading_{i}"] = 0
    scheds = default_schedules[0]
    results = batch_sim(
        features=df,
        timeseries=scheds,
        climate="data/epws/city_epws_indexed/cityidx_0001_USA_NY-New York Central Prk Obs Belv.725033_TMY3.epw",
    )
    print(results)
