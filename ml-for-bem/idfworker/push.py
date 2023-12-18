import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import boto3
import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sqs_queue_if_not_exists(sqs_client, queue_name: str, *, dlq=None):
    try:
        response = sqs_client.create_queue(QueueName=queue_name)
        logger.info(f"Created SQS queue {queue_name}")
        queue_url = response["QueueUrl"]
    except sqs_client.exceptions.QueueNameExists:
        logger.info(f"SQS queue {queue_name} already exists, fetching.")
        queue_url = sqs_client.get_queue_url(QueueName=queue_name)["QueueUrl"]

    if dlq is not None:
        dlq_arn = sqs_client.get_queue_attributes(
            QueueUrl=dlq, AttributeNames=["QueueArn"]
        )["Attributes"]["QueueArn"]
        redrive_policy = {
            "deadLetterTargetArn": dlq_arn,
            "maxReceiveCount": "5",
        }
        sqs_client.set_queue_attributes(
            QueueUrl=queue_url,
            Attributes={"RedrivePolicy": json.dumps(redrive_policy)},
        )
        logger.debug(f"Set DLQ {dlq} for queue {queue_name}")
    return queue_url


def check_if_s3_file_exists(s3_client, bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        logger.debug(f"File {key} exists in bucket {bucket}")
        return True
    except s3_client.exceptions.ClientError:
        logger.debug(f"File {key} does not exist in bucket {bucket}")
        return False


def construct_s3_key(experiment, batch_id, file, *, job_id=None):
    return (
        f"{experiment}/{batch_id}/{os.path.basename(file)}"
        if job_id is None
        else f"{experiment}/{batch_id}/{job_id}/{os.path.basename(file)}"
    )


def upload_to_s3(
    s3_client,
    bucket,
    *,
    key=None,
    experiment=None,
    batch_id=None,
    file_path=None,
):
    assert file_path is not None, "Must specify file_path"
    if key is not None:
        assert (
            experiment is None and batch_id is None
        ), "Cannot specify key and experiment/batch_id"
    else:
        assert (
            experiment is not None and batch_id is not None
        ), "Must specify experiment and batch_id if key is None"
    logger.debug(f"Uploading {file_path} to S3...")
    s3_key = construct_s3_key(experiment, batch_id, file_path) if key is None else key
    s3_client.upload_file(file_path, bucket, s3_key)
    logger.info(f"Uploaded {file_path} to S3")
    return s3_key


def send_to_sqs(sqs_client, queue_url, s3_path, batch_id, experiment):
    logger.info(f"Sending message to SQS queue {queue_url}")
    sqs_client.send_message(
        QueueUrl=queue_url,
        MessageBody=s3_path,
        MessageAttributes={
            "BatchId": {"DataType": "String", "StringValue": batch_id},
            "Experiment": {"DataType": "String", "StringValue": experiment},
        },
    )


@click.command()
@click.option("--bucket", prompt="Bucket name", help="The name of the S3 bucket.")
@click.option(
    "--experiment", prompt="Experiment name", help="The name of the experiment."
)
@click.option("--queue", prompt="Queue name", help="The name of the SQS queue.")
@click.option(
    "--folder", prompt="Folder path", help="The path to the folder with .idf files."
)
@click.option("--epw", prompt="EPW path", help="The path to the EPW file.")
@click.option(
    "--batch_id",
    default=None,
    help="The Batch ID to filter messages.",
    required=False,
)
def push_files(
    bucket,
    experiment,
    queue,
    folder,
    epw,
    batch_id,
):
    sqs_client = boto3.client("sqs")
    s3_client = boto3.client("s3")
    batch_id = str(uuid4()).split("-")[0] if batch_id is None else batch_id
    dlq_queue_url = create_sqs_queue_if_not_exists(sqs_client, queue + "-dlq")
    queue_url = create_sqs_queue_if_not_exists(sqs_client, queue, dlq=dlq_queue_url)

    for file_name in os.listdir(folder):
        if file_name.endswith(".idf") or file_name.endswith(".epjson"):
            job_id = str(uuid4()).split("-")[0]
            idf_file_path = os.path.join(folder, file_name)
            idf_s3_key = upload_to_s3(
                s3_client,
                bucket,
                experiment=experiment,
                batch_id=batch_id + f"/idfs/{job_id}",
                file_path=idf_file_path,
            )
            epw_s3_key = construct_s3_key(experiment, "epw", epw)
            if not check_if_s3_file_exists(s3_client, bucket, epw_s3_key):
                upload_to_s3(s3_client, bucket, key=epw_s3_key, file_path=epw)
            else:
                logger.info(
                    f"Skipping EPW because {epw_s3_key} already exists in bucket {bucket}"
                )
            data = {
                "idf": idf_s3_key,
                "epw": epw_s3_key,
                "bucket": bucket,
                "experiment": experiment,
                "job_id": job_id,
            }
            data = json.dumps(data)
            send_to_sqs(sqs_client, queue_url, data, batch_id, experiment)

    logger.info(f"Batch ID: {batch_id}")


if __name__ == "__main__":
    push_files()
