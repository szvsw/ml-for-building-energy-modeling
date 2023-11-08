import os
from typing import Literal, List
import click
import shutil
import boto3
import pandas as pd
from uuid import uuid4
from utils.nrel_uitls import CLIMATEZONES, RESTYPES
from utils.constants import (
    EPW_MAP_PATH,
    EPW_TESTING_LIST_PATH,
    EPW_TRAINING_LIST_PATH,
    JOULES_TO_KWH,
    EPW_RELATIVE_PATH,
)
import json
from pathlib import Path
import numpy as np
from ladybug.epw import EPW
from schema import Schema, NumericParameter, OneHotParameter, WindowParameter
from shoeboxer.shoebox_config import ShoeboxConfiguration
from shoeboxer.builder import ShoeBox, template_dict
from shoeboxer.schedules import schedules_from_seed, default_schedules
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

data_root = Path("data")

schema = Schema()


testing_epw_df = pd.read_csv(EPW_TESTING_LIST_PATH, index_col=0)
training_epw_df = pd.read_csv(EPW_TRAINING_LIST_PATH, index_col=0)

epw_map = pd.read_csv(EPW_MAP_PATH, index_col=0)


def sample_and_simulate(
    train_or_test: Literal["train", "test"],
    epw_idxs: List[int],
    schedules_mode: Literal["from_seed", "from_idx"],
):
    """
    Sample a building and run a simulation

    Args:
        train_or_test (Literal["train", "test"]): train or test; determines which epw segment to sample
        epw_idxs (List[int], optional): list of epw indices to sample from. Defaults to [];
            samples according to train_or_test if empty.
        schedules_mode: Literal["from_seed", "from_idx"], optional): if "from_seed", then schedules are generated dyanmically.
            Otherwise, schedules are sampled from a pre-generated list.

    Returns:
        sb_name (str): name of shoebox
        simple_dict (dict): dictionary of parameters for space_config.json
        monthly_df (pd.DataFrame): dataframe of monthly results
        err (str): error string
        space_config (dict): dictionary of space config
    """

    """
    Setup
    """
    # reset numpy seed
    np.random.seed()

    """
    Sample Building Parameters
    """
    # Choose random values for storage vector
    parameter_dict = {}
    for param in schema.parameters:
        val = None
        if isinstance(param, OneHotParameter):
            val = np.random.randint(low=0, high=param.count)
        elif isinstance(param, NumericParameter):
            val = np.random.uniform(low=param.min, high=param.max)
        else:
            logger.warning(
                f"Parameter {param.name} is not a sampled parameter type - skipping."
            )
        if val is not None:
            parameter_dict[param.name] = val
    schedules_seed = (
        np.random.randint(1, 2**24 - 1)
        if schedules_mode == "from_seed"
        else np.random.randint(0, default_schedules.shape[0] - 1)
    )
    parameter_dict["schedules_seed"] = schedules_seed

    """
    Sample Shading Vector
    """
    shading_vect = np.random.uniform(0, np.pi / 2 - np.pi / 72, (12,))

    # Check setpoints
    hsp = parameter_dict["HeatingSetpoint"]
    csp = parameter_dict["CoolingSetpoint"]
    if hsp > csp:
        parameter_dict["HeatingSetpoint"] = csp
        parameter_dict["CoolingSetpoint"] = hsp

    """
    Sample/Set Weather/Context Params
    """

    df_epw = training_epw_df if train_or_test == "train" else testing_epw_df
    cz = np.random.choice([cz for cz in df_epw.CZ.unique() if cz != "6C"])
    epw_options = df_epw[df_epw.CZ == cz]
    epw_idx = int(np.random.choice(epw_options.index))

    if len(epw_idxs) > 0:
        epw_idx = int(np.random.choice(epw_idxs))
        cz = epw_map.loc[epw_idx, "CZ"]

    cz_value = CLIMATEZONES[str(cz)]
    parameter_dict["climate_zone"] = cz_value
    parameter_dict["base_epw"] = epw_idx

    """
    Sample/Set Schedules
    """
    schedules_seed = parameter_dict["schedules_seed"]
    schedules = (
        schedules_from_seed(schedules_seed)
        if schedules_mode == "from_seed"
        else default_schedules[schedules_seed]
    )

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
        schedules=schedules,
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
    Make dictionary of parameters for space_config.json
    """
    skip = [
        "batch_id",
        "variation_id",
        "schedules",
    ]
    schema_param_names = [x for x in schema.parameter_names if not x in skip]
    simple_dict = {}

    for param_name in schema_param_names:
        simple_dict[param_name] = parameter_dict[param_name]

    for shading_ix in range(shading_vect.shape[0]):
        simple_dict[f"shading_{shading_ix}"] = shading_vect[shading_ix]

    space_config = {}
    for key in simple_dict:
        param_data = {}
        if "shading" in key:
            param_data["min"] = 0
            param_data["max"] = np.pi / 2
            param_data["mode"] = "Continuous"
            param_data["name"] = key
        elif key in ["schedules_seed", "base_epw", "climate_zone"]:
            continue
        else:
            param = schema[key]
            param_data = {"name": param.name}
            if isinstance(param, NumericParameter):
                param_data["min"] = param.min
                param_data["max"] = param.max
                param_data["mode"] = "Continuous"
            elif isinstance(param, OneHotParameter):
                param_data["option_count"] = param.count
                param_data["mode"] = "Onehot"
        space_config[param_data["name"]] = param_data

    """
    Setup Simulation
    """
    sb_name = str(uuid4())
    output_dir = data_root / "sim_results" / sb_name
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Simulating in {epw_map.loc[epw_idx, 'city']}")
    epw_slug = epw_map.loc[epw_idx, "slug"]
    epw = Path(EPW_RELATIVE_PATH) / epw_slug
    sb = ShoeBox(
        name=sb_name,
        shoebox_config=shoebox_config,
        template_dict=template_datadict,
        epw=epw,
        output_directory=output_dir,
        change_summary=False,
    )

    idf = sb.idf(run_simulation=False)

    """
    Run Simulation
    """
    hourly_df, monthly_df = sb.simulate(idf)
    monthly_df: pd.DataFrame = monthly_df["System"]
    monthly_df = monthly_df.rename(
        columns={
            "PERIMETER IDEAL LOADS AIR": "Perimeter",
            "CORE IDEAL LOADS AIR": "Core",
            "Zone Ideal Loads Supply Air Total Cooling Energy": "Cooling",
            "Zone Ideal Loads Supply Air Total Heating Energy": "Heating",
        },
    )
    monthly_df = monthly_df.unstack()
    monthly_df = monthly_df * JOULES_TO_KWH
    perimeter_area = shoebox_config.width * shoebox_config.perim_depth
    core_area = shoebox_config.width * shoebox_config.core_depth
    monthly_df["Perimeter"] = monthly_df.loc["Perimeter"] / perimeter_area
    monthly_df["Core"] = monthly_df.loc["Core"] / core_area

    """
    Check For Errors
    # TODO: use idf error fetcher from ZLH's work
    """
    errors, warnings = sb.error_report(idf)
    # filter out warnings which we do not care about
    warnings = [
        w
        for w in warnings
        if "Output:Meter: invalid Key Name" not in w
        and "CheckUsedConstructions" not in w
        and "GetPurchasedAir" not in w
        and "The following Report Variables" not in w
        and "psypsatfntemp" not in w.lower()
        and "gpu" not in w.lower()
        and "SizingPeriod:WeatherFileConditionType: object=" not in w
        and "GetSimpleAirModelInputs" not in w
        and "fixviewfactors" not in w.lower()
    ]
    logger.info(f"WARNING COUNT: {len(warnings)}")
    logger.info(f"ERROR COUNT:   {len(errors)}")
    for warning in warnings:
        # these warnnings should be considered hard errors which we want to fully fail on
        if (
            "Standard Time Meridian" in warning
            or "supply humidity ratio" in warning.lower()
            or "not converged" in warning.lower()
        ):
            raise ValueError(warning)
        # other warnings we pass through but will be marked as errors
        logger.warning(warning)
    for error in errors:
        # all errors are hard fails.
        raise ValueError(error)
    err = None
    if len(warnings) > 0 or len(errors) > 0:
        with open(idf.simulation_dir / "eplusout.err", "r") as f:
            err = f.read()
    # # check for errors
    # with open(idf.simulation_dir / "eplusout.end", "r") as f:
    #     summary = f.read()
    #     # Summary format is EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors; Elapsed Time=00hr 00min  0.33sec
    #     # We want to extract the number of warnings and severe errors
    # warnings = int(summary.split("--")[1].split(" ")[1])
    # severe_errors = int(summary.split("--")[1].split(" ")[3])
    # logger.info(f"WARNING COUNT: {warnings}")
    # logger.info(f"ERROR COUNT:   {severe_errors}")
    # err = None
    # if warnings > (19 if os.name == "nt" else 22) or severe_errors > 0:
    #     with open(idf.simulation_dir / "eplusout.err", "r") as f:
    #         err = f.read()

    shutil.rmtree(output_dir)
    return sb_name, simple_dict, monthly_df, err, space_config


def batch_sim(
    n: int,
    train_or_test: Literal["train", "test"] = "train",
    epw_idxs: List[int] = [],
    schedules_mode: Literal["from_seed", "from_idx"] = "from_seed",
):
    """
    Run N simulations and save results to dataframes

    Args:
        n (int): number of simulations to run
        train_or_test (Literal["train", "test"], optional): train or test. Defaults to "train";
            determines which epw segment to sample from as well as which bucket to write to.
        epw_idxs (List[int], optional): list of epw indices to sample from. Defaults to [];
            samples according to train_or_test if empty.
        schedules_mode: Literal["from_seed", "from_idx"], optional): if "from_seed", then schedules are generated dyanmically.
            Otherwise, schedules are sampled from a pre-generated list.

    Returns:
        results (pd.DataFrame): dataframe of results
        err_list (list): list of errors
        space_config (dict): dictionary of space config

    """

    # make a dataframe to store results
    results = pd.DataFrame()

    # make a list to store errors
    err_list = []

    # try_count is used to prevent infinite loops so that we don't
    # blow up our AWS bill if we somehow start failing
    # though we also have a max timeout on the fargate tasks so it's nbd
    try_count = 0

    # run simulations until we have n results
    while len(results) < n:
        # Bail out if needed
        try_count = try_count + 1
        if try_count > 2 * n:
            logger.error("Too many failed simulations! Exiting.")
            break

        # run the sim
        try:
            id, simple_dict, monthly_results, err, space_config = sample_and_simulate(
                train_or_test=train_or_test,
                epw_idxs=epw_idxs,
                schedules_mode=schedules_mode,
            )
        except BaseException as e:
            logger.error("Error during simulation! Continuing.\n\n\n", exc_info=e)
        else:
            logger.info("Finished Simulation!\n\n\n")

            # if we have no results, make a new dataframe
            if len(results) == 0:
                # set the result
                results = pd.DataFrame(monthly_results)
                results = results.T
                # set the index to be a multi index with column names from keys of simple_dict and values from values of simple_dict
                index = (id, *(v for v in simple_dict.values()))
                results.index = pd.MultiIndex.from_tuples(
                    [index],
                    names=["id"] + list(simple_dict.keys()),
                )
            else:
                # make the multi-index of features
                index = (id, *(v for v in simple_dict.values()))
                # set the result
                results.loc[index] = monthly_results
                logger.info(f"Successfully Saved Results! {len(results)}\n\n")
            if err != None:
                # cache any errors
                err_list.append((id, err))

    return results, err_list, space_config


def save_and_upload_batch(batch_dataframe, errs, space_config, bucket, experiment_name):
    """
    Save the batch results to a file and upload to S3

    Args:
        batch_dataframe (pd.DataFrame): dataframe of batch results
        errs (list): list of errors
        bucket (str): S3 bucket name
        experiment_name (str): name of experiment
    """
    # this fargate/batch task has its own id
    run_name = str(uuid4())
    output_folder = data_root / "batch_results"
    os.makedirs(output_folder, exist_ok=True)
    output_path = output_folder / f"{run_name}.hdf"

    batch_dataframe.to_hdf(output_path, key="batch_results")

    logger.info("Connecting to bucket...")
    client = boto3.client("s3")

    logger.info("Uploading Results...")
    client.upload_file(
        Filename=str(output_path),
        Bucket=bucket,
        Key=f"{experiment_name}/monthly/{run_name}.hdf",
    )
    logger.info("Uploading Results Complete")

    if len(errs) > 0:
        logger.info("Uploading Errors...")
        error_root = data_root / "sim_results" / "errors"
        os.makedirs(error_root, exist_ok=True)
        for id, err in errs:
            file_path = error_root / f"{id}.err"
            with open(file_path, "w") as f:
                f.write(err)
            client.upload_file(
                Filename=str(file_path),
                Bucket=bucket,
                Key=f"{experiment_name}/errors/{id}.err",
            )
        logger.info("Uploading Errors Complete")

    # check the aws batch array index
    # only job #1 needs to upload the space config
    array_batch_index = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", None)
    if array_batch_index == 1 or True:
        logger.info("Uploading Space Config...")
        with open(data_root / "space_definition.json", "w") as f:
            json.dump(space_config, f, indent=4)
        client.upload_file(
            Filename=str(data_root / "space_definition.json"),
            Bucket=bucket,
            Key=f"{experiment_name}/space_definition.json",
        )


# make a click function which accepts a number of simulations to run, a bucket name, and an experiment name
@click.command()
@click.option("--n", default=5, help="Number of simulations to run")
@click.option("--bucket", default="ml-for-bem", help="S3 bucket name")
@click.option(
    "--experiment_name", default="single_climate_zone/test", help="Experiment name"
)
# add a click option for train_or_test which must be either "train" or "test"
@click.option("--train_or_test", default="train", help="Train or test")
@click.option(
    "--epw_idxs",
    default=[],
    multiple=True,
    help="epw indices to use (if empty, samples from all climate zones, otherwise, samples from the given list of indices)",
)
@click.option(
    "--schedules_mode",
    default="from_seed",
    help="If 'from_seed', then schedules are generated dyanmically.  Otherwise, schedules are sampled from a pre-generated list.",
)
def main(n, bucket, experiment_name, train_or_test, epw_idxs, schedules_mode):
    batch_dataframe, errs, space_config = batch_sim(
        n, train_or_test, epw_idxs, schedules_mode
    )
    save_and_upload_batch(batch_dataframe, errs, space_config, bucket, experiment_name)


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

    main()
