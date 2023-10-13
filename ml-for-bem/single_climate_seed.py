import os
import click
import shutil
import boto3
import pandas as pd
from uuid import uuid4
from utils.nrel_uitls import CLIMATEZONES, RESTYPES
import json
from pathlib import Path
import numpy as np
from archetypal import UmiTemplateLibrary
from schema import Schema, NumericParameter, OneHotParameter, WindowParameter
from shoeboxer.shoebox_config import ShoeboxConfiguration
from shoeboxer.builder import ShoeBox, template_dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

data_root = Path("data")

with open(data_root / "city_map.json", "r") as f:
    city_map = json.load(f)

schema = Schema()


def sample_and_simulate():
    storage_vector = schema.generate_empty_storage_vector()
    # reset numpy seed
    np.random.seed()
    # Choose random values for storage vector
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
            schema.update_storage_vector(
                storage_vector, parameter=param.name, value=val
            )

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

    cz_string = "4A"
    cz_value = CLIMATEZONES[cz_string]
    schema.update_storage_vector(
        storage_vector, parameter="climate_zone", value=cz_value
    )  # Set all to be run in CZ4
    schema.update_storage_vector(
        storage_vector,
        parameter="program_type",
        value=RESTYPES["Single-Family Detached"],
    )
    schema.update_storage_vector(
        storage_vector,
        parameter="base_epw",
        value=city_map["NY, New York"]["idx"],
    )
    shoebox_config = ShoeboxConfiguration()
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
    shoebox_config.shading_vect = np.zeros((12,))

    # UNCOMMENT TO GET .npy FILE OF RESIDENTIAL TEMPLATE SCHEDULES
    # template_lib_idx = CLIMATEZONES["4A"]

    # cz_value = int(cz_value)

    # template_lib_path = Path(os.getcwd(), "ml-for-bem", "data", "template_libs", "cz_libs", "residential",f"CZ{cz_string}.json")

    # n_masses = 2
    # n_vintages = 4
    # template_idx = (
    #     n_masses * n_vintages * int(RESTYPES["Single-Family Detached"]) + n_masses * 2 + 0
    # )

    # lib = UmiTemplateLibrary.open(template_lib_path)
    # template = lib.BuildingTemplates[template_idx]
    # schedules = schema.parameters[-1].extract_from_template(template)
    # # Save schedules as npy file
    # np.save(Path(os.getcwd(), "ml-for-bem", "data", "residential_schedules.npy"), schedules)

    schedules = np.load(data_root / "residential_schedules.npy")

    skip = [
        "batch_id",
        "variation_id",
        "program_type",
        "vintage",
        "climate_zone",
        "base_epw",
        "schedules_seed",
        "schedules",
        "shading_seed",
    ]
    schema_param_names = [x for x in schema.parameter_names if not x in skip]
    simple_dict = {}
    for param_name in schema_param_names:
        simple_dict[param_name] = schema[param_name].extract_storage_values(
            storage_vector
        )
    wsettings = simple_dict.pop("WindowSettings")
    simple_dict["WindowU"] = wsettings[0]
    simple_dict["WindowSHGC"] = wsettings[1]

    space_config = {}
    for key in simple_dict:
        param_data = {}
        if key == "WindowU":
            wparam: WindowParameter = schema["WindowSettings"]
            param_data = {
                "name": "WindowU",
                "min": wparam.min[0],
                "max": wparam.max[0],
                "mode": "Continuous",
            }
            space_config["WindowU"] = param_data
        elif key == "WindowSHGC":
            wparam: WindowParameter = schema["WindowSettings"]
            param_data = {
                "name": "WindowSHGC",
                "min": wparam.min[1],
                "max": wparam.max[1],
                "mode": "Continuous",
            }
            space_config["WindowSHGC"] = param_data
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
            space_config[param.name] = param_data

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

    # RUNNING TO GET DATAFRAME OF RESULTS
    idf = sb.idf(run_simulation=False)
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

    # check for errors
    with open(idf.simulation_dir / "eplusout.end", "r") as f:
        summary = f.read()
        # Summary format is EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors; Elapsed Time=00hr 00min  0.33sec
        # We want to extract the number of warnings and severe errors
    warnings = int(summary.split("--")[1].split(" ")[1])
    severe_errors = int(summary.split("--")[1].split(" ")[3])
    logger.info(f"WARNING COUNT: {warnings}")
    logger.info(f"ERROR COUNT:   {severe_errors}")
    err = None
    if warnings > (19 if os.name == "nt" else 22) or severe_errors > 0:
        with open(idf.simulation_dir / "eplusout.err", "r") as f:
            err = f.read()

    shutil.rmtree(output_dir)
    return sb_name, simple_dict, monthly_df, err, space_config


def batch_sim(n: int):
    results = pd.DataFrame()
    err_list = []
    try_count = 0
    while len(results) < n:
        try_count = try_count + 1
        if try_count > 2 * n:
            logger.error("Too many failed simulations! Exiting.")
            exit()
        try:
            id, simple_dict, monthly_results, err, space_config = sample_and_simulate()
        except BaseException as e:
            logger.error("Error during simulation! Continuing.\n\n\n", exc_info=e)
        else:
            logger.info("Finished Simulation!\n\n\n")
            if len(results) == 0:
                results = pd.DataFrame(monthly_results)
                results = results.T
                # set the index to be a multi index with column names from keys of simple_dict and values from values of simple_dict
                index = (id, *(v for v in simple_dict.values()))
                results.index = pd.MultiIndex.from_tuples(
                    [index],
                    names=["id"] + list(simple_dict.keys()),
                )
            else:
                index = (id, *(v for v in simple_dict.values()))
                results.loc[index] = monthly_results
                logger.info(f"Successfully Saved Results! {len(results)}\n\n")
            if err != None:
                err_list.append((id, err))
    return results, err_list, space_config


def save_and_upload_batch(batch_dataframe, errs, space_config, bucket, experiment_name):
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
def main(n, bucket, experiment_name):
    batch_dataframe, errs, space_config = batch_sim(n)
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
