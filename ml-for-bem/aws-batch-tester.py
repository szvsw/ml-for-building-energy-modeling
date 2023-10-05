import logging
import click
import os
from datetime import datetime
import h5py
import json
from pathlib import Path

import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import wandb

from storage import upload_to_bucket

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


logging.basicConfig()
logger = logging.getLogger("Sampler Sim Test")
logger.setLevel(logging.INFO)


def config_gcs_adc():
    from storage import creds

    # Copies credentials loaded in from env/json, dump them into local storage file
    with open("credentials.json", "w") as f:
        creds = creds.copy()
        for key, val in creds.items():
            if isinstance(val, bytes):
                val = val.decode("utf-8")

            creds[key] = val
        f.write(json.dumps(creds))

    # Set the credentials env variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"


def make_whitebox_sim():
    # Load in the City Map
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
    schema.update_storage_vector(storage_vector, "floor_2_facade", 0.3)
    schema.update_storage_vector(storage_vector, "core_2_perim", 0.5)
    schema.update_storage_vector(storage_vector, "roof_2_footprint", 0.05)
    schema.update_storage_vector(storage_vector, "ground_2_footprint", 0.5)
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

    return schema, storage_vector


def get_run_config(run, job_ix: int, job_offset: int):
    # TODO: worker count should be passed in as well
    schema = Schema()
    art = run.use_artifact("sample-storage-batches:latest", type="dataset")
    TOTAL_SAMPLES_IN_BATCH = 1000
    JOBS_PER_WORKER = 2
    WORKERS_PER_BATCH = int(TOTAL_SAMPLES_IN_BATCH / JOBS_PER_WORKER)
    batch_id = (job_ix) // WORKERS_PER_BATCH + job_offset
    logger.info(f"Joining queue for {batch_id}...")
    job_in_batch = job_ix % WORKERS_PER_BATCH
    logger.info(f"Worker #{job_in_batch} out of {WORKERS_PER_BATCH}...")
    sample_start_ix = job_in_batch * JOBS_PER_WORKER
    logger.info(
        f"Starting at {sample_start_ix}, will complete {JOBS_PER_WORKER} simulations..."
    )
    # No striding here, just plain stepping
    sample_ixs = np.arange(sample_start_ix, sample_start_ix + JOBS_PER_WORKER, 1)
    # TODO: switch to other sets when batch is above certain threshold
    path = art.get_path(f"train_epws_train_set/BATCH_{batch_id:05d}.hdf5")
    loc = path.download()
    with h5py.File(loc, mode="r") as f:
        storage_batch = f["storage_vectors"][...]
        storage_batch = storage_batch[sample_ixs, :]

    # TODO: make sure you have weather data as needed
    schema.update_storage_batch(
        storage_batch=storage_batch, parameter="base_epw", value=batch_id
    )

    return (
        schema,
        storage_batch,
        batch_id,
        sample_start_ix,
    )


def make_distributed_id(run_name: str, artifact_name: str):
    return f"{run_name}_{artifact_name}"


def make_gcs_upload_dir(artifact_name: str):
    return f"sim-data/{artifact_name}/"


def configure_distributed(
    run_name: str,
    artifact_name: str,
    context: str,
    job_id: int,
    job_offset: int,
    mode: str,
):
    if mode != "finish":
        if context == "LOCAL" and job_id == -1:
            raise ValueError("A Job ID must be provided when running in LOCAL context.")

        aws_job_id = int(os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", -1))
        if context == "AWS":
            if aws_job_id != -1:
                job_id = aws_job_id
            else:
                if job_id == -1:
                    raise ValueError("Running in AWS, but no job_id was provided")

        assert job_id != -1, "No job_id was provided!"
    # Initiate a WandB run for data tracking
    distributed_id = make_distributed_id(
        run_name=run_name,
        artifact_name=artifact_name,
    )

    gcs_upload_dir = make_gcs_upload_dir(artifact_name)
    logger.info(
        f"\nRun Name: {run_name}\nArtifact Name: {artifact_name}\nDistributed ID: {distributed_id}\nJob ID: {job_id}\nJob Offset: {job_offset}"
    )
    return run_name, artifact_name, gcs_upload_dir, distributed_id, job_id, job_offset


def run_distributed(
    run_name: str,
    artifact_name: str,
    context: str,
    job_id: int,
    job_offset: int,
    mode: str,
):
    (
        run_name,
        artifact_name,
        gcs_upload_dir,
        distributed_id,
        job_id,
        job_offset,
    ) = configure_distributed(
        run_name=run_name,
        artifact_name=artifact_name,
        context=context,
        job_id=job_id,
        job_offset=job_offset,
        mode=mode,
    )
    with wandb.init(
        project="ml-for-bem",
        group="batch-simulation",
        job_type="simulation",
        name=run_name,
        save_code=True,
        config={
            "CONTEXT": context,
            "GCS_UPLOAD_DIR": gcs_upload_dir,
            "DISTRIBUTED_ID": distributed_id,
            "ARTIFACT_NAME": artifact_name,
            "JOB_ID": job_id,
            "JOB_OFFSET": job_offset,
        },
    ) as run:
        # Get the job context from AWS Batch

        # Get the vectors to simulate
        (schema, storage_batch, batch_id, sample_start_ix) = get_run_config(
            run, wandb.config.JOB_ID, wandb.config.JOB_OFFSET
        )
        simulations_to_execute = storage_batch.shape[0]

        hourly_hdf5_path = Path(
            f"BATCH_{batch_id:05d}_IX_{sample_start_ix:05d}_hourly.hdf5"
        )
        monthly_hdf5_path = Path(
            f"BATCH_{batch_id:05d}_IX_{sample_start_ix:05d}_monthly.hdf5"
        )

        # Start sequential simulation
        for i in range(simulations_to_execute):
            logger.info(f"---Starting simulation {i}---")

            # Pick out the vector and make whitebox sim
            storage_vector = storage_batch[i]
            try:
                whitebox_sim = WhiteboxSimulation(schema, storage_vector)
            except BaseException as e:
                logger.error(f"ERROR in setting up simulation {i}", exc_info=e)
                continue
            logger.info(whitebox_sim.summarize())

            # Generate results
            try:
                res_hourly, res_monthly = whitebox_sim.simulate()
            except BaseException as e:
                logger.error(f"ERROR in running simulation {i}", exc_info=e)
                continue

            # Save results to local file
            # TODO: concatenate results into a single DF
            res_hourly.to_hdf(
                hourly_hdf5_path,
                key=f"IX_{(i+sample_start_ix):05d}",
                mode="w" if i == 0 else "a",
            )
            res_monthly.to_hdf(
                monthly_hdf5_path,
                key=f"IX_{(i+sample_start_ix):05d}",
                mode="w" if i == 0 else "a",
            )

            with h5py.File(hourly_hdf5_path, mode="r+") as f:
                f.create_dataset(
                    f"storage_vector_{(i+sample_start_ix):05d}",
                    shape=storage_vector.shape,
                    data=storage_vector,
                )
            with h5py.File(monthly_hdf5_path, mode="r+") as f:
                f.create_dataset(
                    f"storage_vector_{(i+sample_start_ix):05d}",
                    shape=storage_vector.shape,
                    data=storage_vector,
                )

        upload_to_bucket(
            blob_name=f"{wandb.config.GCS_UPLOAD_DIR}monthly/{monthly_hdf5_path.name}",
            file_name=monthly_hdf5_path,
        )
        upload_to_bucket(
            blob_name=f"{wandb.config.GCS_UPLOAD_DIR}hourly/{hourly_hdf5_path.name}",
            file_name=hourly_hdf5_path,
        )
        art = wandb.Artifact(
            name=wandb.config.ARTIFACT_NAME,
            type="dataset",
            description="batch simulation results with monthly and hourly files, "
            "which also include storage vectors.  Use pd.HDFStore to read in "
            "'monthly' and 'hourly' dataframes from respective hdf5 files.",
        )
        art.add_reference(
            f"gs://ml-for-bem-data/{wandb.config.GCS_UPLOAD_DIR}", max_objects=50000
        )
        run.upsert_artifact(art, distributed_id=wandb.config.DISTRIBUTED_ID)


def init_distributed(
    run_name: str,
    artifact_name: str,
    context: str,
    job_id: int,
    job_offset: int,
    mode: str,
):
    pass


def finish_distributed(
    run_name: str,
    artifact_name: str,
    context: str,
    job_id: int,
    job_offset: int,
    mode: str,
):
    (
        run_name,
        artifact_name,
        gcs_upload_dir,
        distributed_id,
        job_id,
        job_offset,
    ) = configure_distributed(
        run_name=run_name,
        artifact_name=artifact_name,
        context=context,
        job_id=job_id,
        job_offset=job_offset,
        mode=mode,
    )
    with wandb.init(
        project="ml-for-bem",
        group="batch-simulation",
        job_type="simulation",
        name=run_name,  # TODO: pass this in as an arg
        save_code=True,
        config={
            "GCS_UPLOAD_DIR": gcs_upload_dir,
            "CONTEXT": context,
            "DISTRIBUTED_ID": distributed_id,
            "ARTIFACT_NAME": artifact_name,
            "JOB_ID": job_id,
            "JOB_OFFSET": job_offset,
        },
    ) as run:
        art = wandb.Artifact(
            wandb.config.ARTIFACT_NAME,
            type="dataset",
            description="batch simulation results with monthly and hourly files, "
            "which also include storage vectors.  Use pd.HDFStore to read in "
            "'monthly' and 'hourly' dataframes from respective hdf5 files.",
        )
        art.add_reference(
            f"gs://ml-for-bem-data/{wandb.config.GCS_UPLOAD_DIR}", max_objects=50000
        )
        run.finish_artifact(art)


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["submit", "init", "run", "finish", "test"]),
    required=True,
    help="Operating mode",
)
@click.option(
    "--name",
    type=str,
    required=True,
    help="Name of the run",
)
@click.option(
    "--artifact",
    type=str,
    required=True,
    help="Name of the artifact to generate",
)
@click.option(
    "--context",
    type=click.Choice(["AWS", "LOCAL"]),
    required=True,
    help="Execution context",
)
@click.option(
    "--job_id",
    type=int,
    default=-1,
    required=False,
    help="Optional job id",
)
@click.option(
    "--job_offset",
    type=int,
    required=False,
    default=0,
)
def main(mode, name, artifact, context, job_id, job_offset):
    config_gcs_adc()

    if mode == "test":
        logger.info("TESTING")
        logger.info(f"name: {name}")
        logger.info(f"artifact: {artifact}")
        logger.info(f"context: {context}")
        logger.info(f"job_id: {job_id}")
        logger.info(f"job_offset: {job_offset}")

    if mode == "init":
        init_distributed(
            run_name=name,
            artifact_name=artifact,
            context=context,
            job_id=job_id,
            job_offset=job_offset,
            mode=mode,
        )
    if mode == "run":
        run_distributed(
            run_name=name,
            artifact_name=artifact,
            context=context,
            job_id=job_id,
            job_offset=job_offset,
            mode=mode,
        )
    if mode == "finish":
        finish_distributed(
            run_name=name,
            artifact_name=artifact,
            context=context,
            job_id=job_id,
            job_offset=job_offset,
            mode=mode,
        )


if __name__ == "__main__":
    main()
