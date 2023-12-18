import click
import shutil
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--bucket",
    prompt="Bucket name",
    help="The name of the S3 bucket.",
    default="ml-for-bem",
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
    default="testbatch",
)
@click.option(
    "--dataframe_key",
    prompt="Dataframe key",
    help="The dataframe key to filter messages.",
    default="Buildings",
)
def gather_files(bucket, experiment, batch_id, dataframe_key):
    s3_client = boto3.client("s3")

    batch_path = f"{experiment}/{batch_id}"
    remote_worker_results = f"{batch_path}/results"
    local_batch_path = f"data/worker/{experiment}/{batch_id}"
    worker_file_path = f"{local_batch_path}/gathers"
    local_results_path = f"{local_batch_path}/{dataframe_key}.hdf"
    remote_upload_path = f"{batch_path}/{dataframe_key}.hdf"
    os.makedirs(worker_file_path, exist_ok=True)

    # get all files in batch results folder using a paginator
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=remote_worker_results)
    files = []
    for page in page_iterator:
        if "Contents" in page:
            for key in page["Contents"]:
                files.append(key["Key"])

    logger.info(f"Found {len(files)} files in {remote_worker_results}")

    def download_and_open(key):
        file_name = key.split("/")[-1]
        local_file_path = f"{worker_file_path}/{file_name}"
        s3_client.download_file(bucket, key, local_file_path)
        df = pd.read_hdf(local_file_path, key=dataframe_key)
        os.remove(local_file_path)
        return df

    logger.info("Downloading and opening files...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        dfs = list(tqdm(executor.map(download_and_open, files), total=len(files)))
    logger.info("Downloading and opening files complete.")

    logger.info("Concatenating dataframes...")
    df = pd.concat(dfs, axis=0)
    logger.info("Concatenating dataframes complete.")

    logger.info("Saving file...")
    df.to_hdf(local_results_path, key=dataframe_key)
    logger.info("Saving file complete.")

    logger.info("Uploading file...")
    s3_client.upload_file(local_results_path, bucket, remote_upload_path)
    logger.info("Uploading file complete.")

    shutil.rmtree(local_batch_path)


if __name__ == "__main__":
    gather_files()
