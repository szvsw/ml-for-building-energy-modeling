import boto3
import click
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def receive_n_messages(
    sqs_client,
    queue_url,
    num_msgs=1,
    wait_time=1,
    visibility_timeout=120,
):
    """
    Receive messages from SQS queue.

    Args:
        sqs_client: The SQS client.
        queue_url: The URL of the SQS queue.
        num_msgs: The number of messages to receive (default: 1).
        wait_time: The time in seconds to wait for messages (default: 1).
        visibility_timeout: The time in seconds to make received messages
            invisible to other consumers (default: 120).

    Returns:
        The SQS response.
    """
    return sqs_client.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=num_msgs,
        WaitTimeSeconds=wait_time,
        VisibilityTimeout=visibility_timeout,
        MessageAttributeNames=["All"],
    )


def delete_message(sqs_client, queue_url, receipt_handle):
    """
    Delete message from SQS queue.

    Args:
        sqs_client: The SQS client.
        queue_url: The URL of the SQS queue.
        receipt_handle: The receipt handle of the message to delete.

    Returns:
        None
    """
    sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)


def reset_message_visibility(
    sqs_client, queue_url, receipt_handle, visibility_timeout=120
):
    """
    Reset message visibility timeout.

    Args:
        sqs_client: The SQS client.
        queue_url: The URL of the SQS queue.
        receipt_handle: The receipt handle of the message to reset.
        visibility_timeout: The time in seconds to make received messages
            invisible to other consumers (default: 120).

    Returns:
        None
    """

    sqs_client.change_message_visibility(
        QueueUrl=queue_url,
        ReceiptHandle=receipt_handle,
        VisibilityTimeout=visibility_timeout,
    )


def process_message(message, handler=None):
    """
    Process message.

    Args:
        message: The message to process.
        handler (callable, default=None): The function to handle the message.  If none, just prints the message.

    Returns:
        The result of the handler (or None if not specified).
    """
    if handler:
        return handler(message)
    else:
        logger.warning(f"No handler specified for message:")
        logger.warning(message)
        return None


@click.command()
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
)
@click.option(
    "--batch_id",
    prompt="Batch ID",
    help="The Batch ID to filter messages.",
    default="idf/batch/test",
)
@click.option(
    "--num_messages_to_process",
    prompt="Number of messages",
    help="Number of messages.",
    default=10,
)
@click.option(
    "--visibility_timeout",
    default=120,
    help="Visibility timeout in seconds.",
)
@click.option(
    "--wait_time",
    default=1,
    help="Time in seconds to wait for messages to arrive.",
)
@click.option(
    "--num_msgs_per_request",
    default=10,
    help="Number of messages to receive per queue request.",
)
def consume_messages(
    queue,
    experiment,
    batch_id,
    num_messages_to_process,
    num_msgs_per_request,
    visibility_timeout,
    wait_time,
    handler=None,
):
    sqs_client = boto3.client("sqs")
    queue_url = sqs_client.get_queue_url(QueueName=queue)["QueueUrl"]

    processed_messages = 0
    attempts_made = 0
    results = []

    while processed_messages < int(num_messages_to_process) and attempts_made < 3 * int(
        num_messages_to_process
    ):
        logger.info(f"--- Attempt {attempts_made:03d} ---")
        logger.info(f"fetching messages from SQS")
        response = receive_n_messages(
            sqs_client=sqs_client,
            queue_url=queue_url,
            num_msgs=num_msgs_per_request,
            wait_time=wait_time,
            visibility_timeout=visibility_timeout,
        )
        messages = response.get("Messages", [])
        logger.info(f"Received {len(messages)} messages from SQS queue {queue_url}")

        for msg in messages:
            attrs = msg.get("MessageAttributes", {})
            if (
                attrs.get("Experiment", {}).get("StringValue") == experiment
                and attrs.get("BatchId", {}).get("StringValue") == batch_id
            ):
                logger.info(f"Processing message: {msg['MessageId']}")
                result = process_message(message=msg["Body"], handler=handler)
                results.append(result)
                delete_message(
                    sqs_client=sqs_client,
                    queue_url=queue_url,
                    receipt_handle=msg["ReceiptHandle"],
                )
                processed_messages += 1
                logger.info(f"Processed {processed_messages} messages")
            else:
                logger.warning(f"Skipping message: {msg['MessageId']}")
                logger.warning(
                    f"Expected Experiment: {experiment}, but got {attrs.get('Experiment', {}).get('StringValue')}"
                )
                logger.warning(
                    f"Expected BatchId: {batch_id}, but got {attrs.get('BatchId', {}).get('StringValue')}"
                )
                reset_message_visibility(
                    sqs_client=sqs_client,
                    queue_url=queue_url,
                    receipt_handle=msg["ReceiptHandle"],
                    visibility_timeout=0,
                )
            attempts_made += 1
    logger.info(
        f"Finished processing messages, processed {processed_messages} messages"
    )

    return results


if __name__ == "__main__":
    consume_messages()
