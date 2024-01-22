import json
import logging
import os
import shutil
from functools import partial
from pathlib import Path
from uuid import UUID, uuid4

import boto3
import click
import pandas as pd
from archetypal.idfclass import IDF
from archetypal.idfclass.sql import Sql
from idfworker.pull import consume_messages
from idfworker.push import construct_s3_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def construct_local_data_path(worker_id: str) -> Path:
    data_path = Path("./") / "data" / "worker" / worker_id
    os.makedirs(data_path, exist_ok=True)
    return data_path


def download_s3_file(s3_client, bucket, key, local_path):
    if not os.path.exists(local_path):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {key} from S3...")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {key} from S3")
    else:
        logger.info(f"File {local_path} already exists, skipping download.")


def handler(*, s3_client, worker_id: UUID, data_path: Path, message: dict):
    data = json.loads(message["Body"])
    logger.info(f"Worker ID: {worker_id}")
    logger.debug(f"Received message: {data}")
    logger.debug(f"Message ID: {message['MessageId']}")
    logger.debug(f"Receipt handle: {message['ReceiptHandle']}")
    bucket = data["bucket"]
    idf_key = data["idf"]
    epw_key = data["epw"]
    job_id = data["job_id"]
    experiment = data["experiment"]
    local_idf_path = data_path / idf_key
    local_epw_path = data_path / epw_key  # only download if necessary
    output_dir = local_idf_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    download_s3_file(s3_client, bucket, idf_key, local_idf_path)
    download_s3_file(s3_client, bucket, epw_key, local_epw_path)

    idf = IDF(
        idfname=local_idf_path,
        epw=local_epw_path,
        output_directory=output_dir,
    )
    idf.simulate()
    sql = Sql(idf.sql_file)
    zone_monthly_df = pd.DataFrame(
        sql.timeseries_by_name(
            variable_or_meter=[
                "Zone Ideal Loads Supply Air Total Heating Energy",
                "Zone Ideal Loads Supply Air Total Cooling Energy",
            ],
            reporting_frequency="Monthly",
        )
    )
    zone_monthly_df = postprocess(zone_monthly_df, local_idf_path, job_id)

    remote_results_key = Path(idf_key).with_suffix(".hdf")
    local_results_key = local_idf_path.with_suffix(".hdf")
    zone_monthly_df.to_hdf(local_results_key, key="Zones")

    building_monthly_df = pd.DataFrame(
        sql.timeseries_by_name(
            variable_or_meter=[
                "Heating:DistrictHeating",
                "Cooling:DistrictCooling",
                "Electricity:Facility",
            ],
            reporting_frequency="Monthly",
        )
    )
    # drop the first two levels of the columns
    building_monthly_df.columns = building_monthly_df.columns.droplevel(level=[0, 1])
    building_monthly_df = building_monthly_df.unstack()
    building_monthly_df.to_hdf(local_results_key, key="Building", mode="a")

    s3_client.upload_file(
        str(local_results_key),
        bucket,
        str(remote_results_key),
    )

    return (
        zone_monthly_df,
        building_monthly_df,
        (job_id, local_idf_path.name, local_epw_path.name),
    )


def postprocess(results_df: pd.DataFrame, local_idf_path, job_id):
    df: pd.DataFrame = results_df["System"]

    col_renamer = {
        "Zone Ideal Loads Supply Air Total Cooling Energy": "Cooling",
        "Zone Ideal Loads Supply Air Total Heating Energy": "Heating",
    }
    df = df.rename(
        columns=col_renamer,
    )
    new_cols = []
    for col in df.columns:
        zone_name = col[0].split(" ")[0]
        new_cols.append((zone_name, col[1]))
    df.columns = pd.MultiIndex.from_tuples(new_cols)
    df.index.name = "Month"
    df = df.stack(level=0)
    df = df.unstack(level=0)
    df.index = pd.MultiIndex.from_tuples(
        [(job_id, local_idf_path.name, zone) for zone in df.index.to_list()],
        names=["job_id", "file_name", "Zone"],
    )
    df.columns.names = ["EndUse", "Month"]
    return df


@click.command()
@click.option(
    "--bucket",
    prompt="Bucket name",
    help="The name of the S3 bucket.",
    default="ml-for-bem",
)
@click.option(
    "--queue",
    prompt="Queue name",
    help="The name of the SQS queue.",
    default="ml-for-bem,",
)
@click.option(
    "--experiment",
    prompt="Experiment name",
    help="The name of the experiment.",
    default="idf/batch/test",
)
@click.option(
    "--batch_id",
    prompt="Batch ID",
    help="The Batch ID to filter messages.",
    default="test",
)
@click.option(
    "--num_messages_to_process",
    prompt="Number of messages",
    help="Number of messages.",
    type=int,
    default=10,
)
@click.option(
    "--visibility_timeout",
    default=120,
    type=int,
    help="Visibility timeout in seconds.",
)
@click.option(
    "--wait_time",
    default=0,
    type=int,
    help="Time in seconds to wait for messages to arrive.",
)
@click.option(
    "--num_messages_per_request",
    default=1,
    type=int,
    help="Number of messages to receive per queue request.",
)
@click.option(
    "--dry_run",
    default=False,
    help="Dry run.",
)
@click.option(
    "--worker_id",
    default=None,
    help="The worker ID.",
)
def run(
    bucket,
    queue,
    experiment,
    batch_id,
    num_messages_to_process,
    visibility_timeout,
    wait_time,
    num_messages_per_request,
    dry_run,
    worker_id,
):
    """
    Run the worker.

    Args:
        queue (str, default="ml-for-bem"): The name of the SQS queue.
        experiment (str, default="idf/batch/test"): The name of the experiment.
        batch_id (str, default="test"): The Batch ID to filter messages.
        num_messages_to_process (int, default=10): The number of messages to process.
        visibility_timeout (int, default=120): The visibility timeout in seconds.
        wait_time (float, default=1): The time in seconds to wait for messages to arrive.
        num_messages_per_request (int, default=1): The number of messages to receive per queue request.
        dry_run (bool, default=False): Dry run.
        worker_id (UUID, default=None): The worker ID.
    """
    s3_client = boto3.client("s3")
    worker_id = str(uuid4()).split("-")[0] if worker_id is None else worker_id
    data_path = construct_local_data_path(worker_id)

    msg_handler = partial(
        handler, s3_client=s3_client, worker_id=worker_id, data_path=data_path
    )

    msg_handler = msg_handler if not dry_run else lambda msg: logger.info(msg["Body"])

    results = consume_messages(
        queue=queue,
        experiment=experiment,
        batch_id=batch_id,
        num_messages_to_process=num_messages_to_process,
        num_msgs_per_request=num_messages_per_request,
        visibility_timeout=visibility_timeout,
        wait_time=wait_time,
        handler=msg_handler,
    )
    if dry_run:
        return

    zone_results_to_concat = [
        result[0] for result in results if isinstance(result, tuple)
    ]
    building_results_to_concat = [
        result[1] for result in results if isinstance(result, tuple)
    ]
    building_idxs = [result[2] for result in results if isinstance(result, tuple)]

    if len(zone_results_to_concat) == 0:
        logger.info("No results to concatenate.")
        return

    zone_results = pd.concat(zone_results_to_concat, axis=0)

    buildings_df = pd.DataFrame()
    for idx, building_df in zip(building_idxs, building_results_to_concat):
        if len(buildings_df) == 0:
            buildings_df = pd.DataFrame(building_df)
            buildings_df = buildings_df.T
            buildings_df.index = pd.MultiIndex.from_tuples(
                [idx], names=["job_id", "file_name", "epw_name"]
            )
        else:
            buildings_df.loc[idx] = building_df

    results_local_path = data_path / f"{worker_id}.hdf"
    logger.info("Saving results...")
    zone_results.to_hdf(results_local_path, key="Zones")
    buildings_df.to_hdf(results_local_path, key="Buildings", mode="a")
    logger.info("Uploading results to S3...")
    results_key = construct_s3_key(
        experiment=experiment,
        batch_id=batch_id,
        file=results_local_path,
        job_id="results",
    )
    s3_client.upload_file(
        str(results_local_path),
        bucket,
        results_key,
    )
    shutil.rmtree(data_path)
    logger.info("Uploaded results to S3.")


if __name__ == "__main__":
    from archetypal import settings

    # Check if we are running on Windows or Linux using os
    if os.name == "nt":
        settings.ep_version == "22.2.0"
        settings.energyplus_location = Path("C:/EnergyPlusV22-2-0")
    else:
        settings.ep_version == "22.2.0"
        settings.energyplus_location = Path("/usr/local/EnergyPlus-22-2-0")

    run()
