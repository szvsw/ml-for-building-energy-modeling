import boto3
import click
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def receive_messages(queue_url, num_msgs=1, wait_time=1, visibility_timeout=120):
    """
    Receive messages from SQS queue.

    Args:
        queue_url: The URL of the SQS queue.
        num_msgs: The number of messages to receive (default: 10).
        wait_time: The time in seconds to wait for messages (default: 20).
        visibility_timeout: The time in seconds to make received messages
            invisible to other consumers (default: 30).
    """
    sqs_client = boto3.client("sqs")
    return sqs_client.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=num_msgs,
        WaitTimeSeconds=wait_time,
        VisibilityTimeout=visibility_timeout,
        MessageAttributeNames=["All"],
    )


def delete_message(queue_url, receipt_handle):
    sqs_client = boto3.client("sqs")
    sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)


def process_message(message):
    # Placeholder for processing logic
    print("Processing message:", message)


@click.command()
@click.option("--bucket", prompt="Bucket name", help="The name of the S3 bucket.")
@click.option("--queue", prompt="Queue name", help="The name of the SQS queue.")
@click.option(
    "--experiment", prompt="Experiment name", help="The name of the experiment."
)
@click.option("--batch_id", prompt="Batch ID", help="The Batch ID to filter messages.")
@click.option("--n_messages", prompt="Number of messages", help="Number of messages.")
def consume_messages(bucket, queue, experiment, batch_id, n_messages):
    sqs_client = boto3.client("sqs")
    queue_url = sqs_client.get_queue_url(QueueName=queue)["QueueUrl"]

    for i in range(int(n_messages)):
        logger.info(f"{i:03d}: fetching messages from SQS")
        response = receive_messages(queue_url)
        messages = response.get("Messages", [])
        logger.info(f"Received {len(messages)} messages from SQS queue {queue_url}")

        for msg in messages:
            attrs = msg.get("MessageAttributes", {})
            if (
                attrs.get("Experiment", {}).get("StringValue") == experiment
                and attrs.get("BatchId", {}).get("StringValue") == batch_id
            ):
                logger.info(f"Processing message: {msg['MessageId']}")
                process_message(msg["Body"])
                delete_message(queue_url, msg["ReceiptHandle"])
            else:
                logger.warning(f"Skipping message: {msg['MessageId']}")
                logger.warning(
                    f"Expected Experiment: {experiment}, but got {attrs.get('Experiment', {}).get('StringValue')}"
                )
                logger.warning(
                    f"Expected BatchId: {batch_id}, but got {attrs.get('BatchId', {}).get('StringValue')}"
                )


if __name__ == "__main__":
    consume_messages()
