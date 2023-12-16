import os
import uuid
import boto3
import click


def generate_uuid():
    return str(uuid.uuid4())


def create_sqs_queue_if_not_exists(sqs_client, queue_name):
    try:
        response = sqs_client.create_queue(QueueName=queue_name)
        return response["QueueUrl"]
    except sqs_client.exceptions.QueueNameExists:
        return sqs_client.get_queue_url(QueueName=queue_name)["QueueUrl"]


def upload_to_s3(bucket, experiment, batch_id, file_path):
    s3_client = boto3.client("s3")
    file_name = os.path.basename(file_path)
    s3_key = f"{experiment}/{batch_id}/{file_name}"
    s3_client.upload_file(file_path, bucket, s3_key)
    return f"s3://{bucket}/{s3_key}"


def send_to_sqs(queue_url, s3_path, batch_id, experiment):
    sqs_client = boto3.client("sqs")
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
def process_files(
    bucket,
    experiment,
    queue,
    folder,
):
    batch_id = generate_uuid()
    sqs_client = boto3.client("sqs")
    queue_url = create_sqs_queue_if_not_exists(sqs_client, queue)

    for file_name in os.listdir(folder):
        if file_name.endswith(".idf"):
            file_path = os.path.join(folder, file_name)
            s3_path = upload_to_s3(bucket, experiment, batch_id, file_path)
            send_to_sqs(queue_url, s3_path, batch_id, experiment)

    print(f"Batch ID: {batch_id}")


if __name__ == "__main__":
    """
    To Run:
    python orchestrate.py --bucket <bucket_name> --experiment <experiment_name> --queue <queue_name> --folder <folder_path>
    """
    process_files()
