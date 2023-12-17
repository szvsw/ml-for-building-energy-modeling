import os
import logging
import json
import uuid
import boto3
import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_uuid():
    return str(uuid.uuid4())


def create_sqs_queue_if_not_exists(sqs_client, queue_name):
    try:
        response = sqs_client.create_queue(QueueName=queue_name)
        logger.info(f"Created SQS queue {queue_name}")
        return response["QueueUrl"]
    except sqs_client.exceptions.QueueNameExists:
        logger.info(f"SQS queue {queue_name} already exists, fetching.")
        return sqs_client.get_queue_url(QueueName=queue_name)["QueueUrl"]


def check_if_s3_file_exists(s3_client, bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        logger.info(f"File {key} exists in bucket {bucket}")
        return True
    except s3_client.exceptions.ClientError:
        logger.info(f"File {key} does not exist in bucket {bucket}")
        return False


def construct_s3_key(experiment, batch_id, file):
    return f"{experiment}/{batch_id}/{os.path.basename(file)}"


def upload_to_s3(s3_client, bucket, experiment, batch_id, file_path):
    logger.info(f"Uploading {file_path} to S3...")
    s3_key = construct_s3_key(experiment, batch_id, file_path)
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
def process_files(
    bucket,
    experiment,
    queue,
    folder,
    epw,
    batch_id,
):
    sqs_client = boto3.client("sqs")
    s3_client = boto3.client("s3")
    batch_id = generate_uuid() if batch_id is None else batch_id
    queue_url = create_sqs_queue_if_not_exists(sqs_client, queue)

    for file_name in os.listdir(folder):
        if file_name.endswith(".idf") or file_name.endswith(".epjson"):
            idf_file_path = os.path.join(folder, file_name)
            idf_s3_path = upload_to_s3(
                s3_client, bucket, experiment, batch_id, idf_file_path
            )
            if not check_if_s3_file_exists(
                s3_client, bucket, construct_s3_key(experiment, batch_id, epw)
            ):
                epw_s3_path = upload_to_s3(s3_client, bucket, experiment, batch_id, epw)
            else:
                epw_s3_path = construct_s3_key(experiment, batch_id, epw)
            data = {"idf": idf_s3_path, "epw": epw_s3_path, "bucket": bucket}
            data = json.dumps(data)
            send_to_sqs(sqs_client, queue_url, data, batch_id, experiment)

    print(f"Batch ID: {batch_id}")


if __name__ == "__main__":
    process_files()
