import os
import json
import shutil
import logging

from pathlib import Path

from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig()
logger = logging.getLogger("Storage")
logger.setLevel(logging.INFO)


try:
    creds = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    storage_client = storage.Client.from_service_account_info(creds)
    logger.info("Successfully opened GCS client.")
except BaseException as e:
    logger.error("ERR", exc_info=e)
    logger.warning(
        "Could not find GCS credentials in system env, falling back to json."
    )
    storage_client = storage.Client.from_service_account_json(
        Path(os.path.dirname(os.path.abspath(__file__)))
        / ".."
        / "credentials"
        / "bucket-key.json"
    )

try:
    bucket = storage_client.get_bucket("ml-for-bem-data")
    logger.info("Successfully fetched bucket location")
except BaseException as e:
    logger.error("Error fetching bucket location", exc_info=e)
    raise e


def check_bucket_completeness():
    found = []
    for blob in storage_client.list_blobs("ml-for-bem-data", prefix="final_results"):
        for i in range(591):
            if f"{i:05d}" in str(blob):
                found.append(i)
                break
    missing = []
    for i in range(591):
        if i not in found:
            print(f"Batch {i:05d} is missing.")
            missing.append(i)
    return missing


def upload_to_bucket(blob_name, file_name):
    logger.info(f"Uploading {file_name} to bucket:{blob_name}...")
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_name)
    logger.info(f"Done uploading.")


def download_from_bucket(blob_name, file_name):
    logger.info(f"Downloading bucket:{blob_name} to file:{file_name}...")
    blob = bucket.blob(blob_name)
    blob.download_to_filename(file_name)
    logger.info(f"Done downloading.")


def download_batches(prefix="final_results"):
    os.makedirs("./data/hdf5/" + prefix, exist_ok=True)
    for blob in storage_client.list_blobs("ml-for-bem-data", prefix=prefix):
        logger.info(f"Downloading {blob.name}")
        blob.download_to_filename("data/hdf5/" + blob.name)
        logger.info(f"Finshed downloading {blob.name}")


def download_epws():
    zip_path = (
        Path(os.path.dirname(os.path.abspath(__file__)))
        / "data"
        / "epws"
        / "global_epws_indexed.zip"
    )
    unzip_folder = (
        Path(os.path.dirname(os.path.abspath(__file__)))
        / "data"
        / "epws"
        / "global_epws_indexed"
    )
    download_from_bucket("epws/global_epws_indexed.zip", zip_path)
    logger.info("Unzipping EPWs...")
    os.makedirs(unzip_folder, exist_ok=True)
    shutil.unpack_archive(zip_path, unzip_folder)
    logger.info("Done unzipping EPWs.")


if __name__ == "__main__":
    # download_epws()
    check_bucket_completeness()
