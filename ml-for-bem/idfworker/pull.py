import logging

import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_message(*, sqs_client, queue_url, message, handler=None):
    """
    Process message.

    Args:
        sqs_client: The SQS client.
        queue_url: The SQS queue URL.
        message: The message to process.
        handler (callable, default=None): The function to handle the message.  If none, just prints the message.

    Returns:
        The result of the handler (or None if not specified).
    """
    if handler:
        if callable(handler):
            try:
                return handler(message=message)
            except Exception as e:
                logger.error(f"Error processing message: {message['Body']}", exc_info=e)
                sqs_client.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=message["ReceiptHandle"],
                    VisibilityTimeout=0,
                )
                return 1
        else:
            raise TypeError(f"handler must be callable, not {type(handler)}")
    else:
        logger.warning(f"No handler specified for message:")
        logger.warning(message["Body"])
        return None


def consume_messages(
    *,
    queue,
    experiment,
    batch_id,
    num_messages_to_process,
    num_msgs_per_request,
    visibility_timeout,
    wait_time,
    handler,
):
    """
    Consume messages from SQS queue.

    Args:
        queue: The name of the SQS queue.
        experiment: The name of the experiment.
        batch_id: The Batch ID to filter messages.
        num_messages_to_process: The number of messages to process.
        num_msgs_per_request: The number of messages to receive per queue request.
        visibility_timeout: The visibility timeout in seconds.
        wait_time: The time in seconds to wait for messages to arrive.
        handler (callable): The function to handle the message.
    """
    sqs_client = boto3.client("sqs")
    queue_url = sqs_client.get_queue_url(QueueName=queue)["QueueUrl"]

    processed_messages = 0
    attempts_made = 0
    results = []

    while processed_messages < int(num_messages_to_process) and attempts_made < 3 * int(
        num_messages_to_process
    ):
        logger.info("")
        logger.info(f"--- Attempt {attempts_made:03d} ---")
        response = sqs_client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=num_msgs_per_request,
            WaitTimeSeconds=wait_time,
            VisibilityTimeout=visibility_timeout,
            MessageAttributeNames=["All"],
        )

        messages = response.get("Messages", [])
        logger.debug(f"{len(messages)} messages fetched from SQS queue {queue_url}")
        if len(messages) == 0:
            logger.debug("No messages received, continuing...")
            attempts_made += 1
            continue

        for msg in messages:
            attrs = msg.get("MessageAttributes", {})
            if processed_messages >= int(num_messages_to_process):
                logger.info("Skipping message, already processed enough messages")
                sqs_client.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=msg["ReceiptHandle"],
                    VisibilityTimeout=0,
                )
            if (
                attrs.get("Experiment", {}).get("StringValue") == experiment
                and attrs.get("BatchId", {}).get("StringValue") == batch_id
            ):
                logger.debug(f"Processing message: {msg['MessageId']}")
                result = process_message(
                    sqs_client=sqs_client,
                    queue_url=queue_url,
                    message=msg,
                    handler=handler,
                )
                if result is 1:
                    results.append(None)
                    logger.debug("Message processing failed, adding back into queue...")
                else:
                    results.append(result)
                    sqs_client.delete_message(
                        QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"]
                    )
                    processed_messages += 1
                    logger.debug(f"Processed {processed_messages} messages")
            else:
                logger.info(f"Skipping message: {msg['MessageId']}")
                logger.debug(
                    f"Expected Experiment: {experiment}, but got {attrs.get('Experiment', {}).get('StringValue')}"
                )
                logger.debug(
                    f"Expected BatchId: {batch_id}, but got {attrs.get('BatchId', {}).get('StringValue')}"
                )

                sqs_client.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=msg["ReceiptHandle"],
                    VisibilityTimeout=0,
                )
            attempts_made += 1
    logger.info("")
    logger.info(
        f"Finished processing messages, processed {processed_messages} messages\n"
    )

    return results


if __name__ == "__main__":
    consume_messages()
