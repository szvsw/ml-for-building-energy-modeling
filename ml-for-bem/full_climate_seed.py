import os
from typing import Literal
import click
import shutil
import boto3
import pandas as pd
from uuid import uuid4
from utils.nrel_uitls import CLIMATEZONES, RESTYPES
from utils.constants import EPW_MAP_PATH, EPW_TESTING_LIST_PATH, EPW_TRAINING_LIST_PATH
import json
from pathlib import Path
import numpy as np
from archetypal import UmiTemplateLibrary
from schema import Schema, NumericParameter, OneHotParameter, WindowParameter
from shoeboxer.shoebox_config import ShoeboxConfiguration
from shoeboxer.builder import ShoeBox, template_dict, schedules_from_seed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

data_root = Path("data")

schema = Schema()


testing_epw_df = pd.read_csv(EPW_TESTING_LIST_PATH, index_col=0)
training_epw_df = pd.read_csv(EPW_TRAINING_LIST_PATH, index_col=0)


def sample_and_simulate(train_or_test: Literal["train", "test"]):
    """
    Sample a building and run a simulation

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
    storage_vector = schema.generate_empty_storage_vector()
    # reset numpy seed
    np.random.seed()

    """
    Sample Building Parameters
    """
    # Choose random values for storage vector
    for param in schema.parameters:
        val = None
        if isinstance(param, OneHotParameter):
            val = np.random.randint(low=0, high=param.count)
        elif isinstance(param, NumericParameter):
            # TODO: check window parameters sampling occurs separately
            val = np.random.uniform(low=param.min, high=param.max)
        else:
            logger.warning(
                f"Parameter {param.name} is not a sampled parameter type - skipping."
            )
        if val is not None:
            schema.update_storage_vector(
                storage_vector, parameter=param.name, value=val
            )
    schedules_seed = np.random.randint(1, 2**24 - 1)
    schema.update_storage_vector(
        storage_vector, parameter="schedules_seed", value=schedules_seed
    )

    """
    Sample Shading Vector
    """
    shading_vect = np.random.uniform(0, np.pi / 2 - np.pi / 72, (12,))

    # Check setpoints
    hsp = schema["HeatingSetpoint"].extract_storage_values(storage_vector)
    csp = schema["CoolingSetpoint"].extract_storage_values(storage_vector)
    if hsp > csp:
        schema.update_storage_vector(
            storage_vector, parameter="CoolingSetpoint", value=hsp
        )
        schema.update_storage_vector(
            storage_vector, parameter="HeatingSetpoint", value=csp
        )

    """
    Sample/Set Weather/Context Params
    """

    df_epw = training_epw_df if train_or_test == "train" else testing_epw_df
    cz = np.random.choice(df_epw.CZ.unique())
    epw_options = df_epw[df_epw.CZ == cz]
    epw_idx = int(np.random.choice(epw_options.index))

    cz_value = CLIMATEZONES[str(cz)]
    schema.update_storage_vector(
        storage_vector,
        parameter="climate_zone",
        value=cz_value,
    )
    schema.update_storage_vector(
        storage_vector,
        parameter="base_epw",
        value=epw_idx,
    )

    """
    Make the Shoebox Config
    """
    shoebox_config = ShoeboxConfiguration()

    """
    Build the Shoebox
    """
    shoebox_config.shading_vect = shading_vect
    shoebox_config.width = schema["width"].extract_storage_values(storage_vector)
    shoebox_config.height = schema["height"].extract_storage_values(storage_vector)
    shoebox_config.floor_2_facade = schema["floor_2_facade"].extract_storage_values(
        storage_vector
    )
    shoebox_config.core_2_perim = schema["core_2_perim"].extract_storage_values(
        storage_vector
    )
    shoebox_config.roof_2_footprint = schema["roof_2_footprint"].extract_storage_values(
        storage_vector
    )
    shoebox_config.ground_2_footprint = schema[
        "ground_2_footprint"
    ].extract_storage_values(storage_vector)
    shoebox_config.wwr = schema["wwr"].extract_storage_values(storage_vector)
    shoebox_config.orientation = schema["orientation"].extract_storage_values(
        storage_vector
    )

    """
    Sample/Set Schedules
    """
    schedules_seed = schema["schedules_seed"].extract_storage_values(storage_vector)
    schedules = schedules_from_seed(schedules_seed)
    """
    Make dictionary of parameters for space_config.json
    """
    skip = [
        "batch_id",
        "variation_id",
        "schedules",
        "shading_seed",
    ]
    schema_param_names = [x for x in schema.parameter_names if not x in skip]
    simple_dict = {}
    for param_name in schema_param_names:
        simple_dict[param_name] = schema[param_name].extract_storage_values(
            storage_vector
        )
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
    sb = ShoeBox.from_vector(
        name=sb_name,
        schema=schema,
        shoebox_config=shoebox_config,
        vector=storage_vector,
        schedules=schedules,
        change_summary=False,
        output_directory=output_dir,
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

    """
    Check For Errors
    # TODO: use idf error fetcher from ZLH's work
    """
    errors, warnings = sb.error_report(idf)
    warnings = [
        w
        for w in warnings
        if "Output:Meter: invalid Key Name" not in w
        and "CheckUsedConstructions" not in w
        and "GetPurchasedAir" not in w
        and "The following Report Variables" not in w
        and "psysatfntemp" not in w.lower()
        and "gpu" not in w.lower()
    ]
    logger.info(f"WARNING COUNT: {len(warnings)}")
    logger.info(f"ERROR COUNT:   {len(errors)}")
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


def batch_sim(n: int, train_or_test: Literal["train", "test"] = "train"):
    """
    Run N simulations and save results to dataframes

    Args:
        n (int): number of simulations to run

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
            exit()

        # run the sim
        try:
            id, simple_dict, monthly_results, err, space_config = sample_and_simulate(
                train_or_test=train_or_test
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
def main(n, bucket, experiment_name, train_or_test):
    batch_dataframe, errs, space_config = batch_sim(n, train_or_test)
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
