import logging
import click
import json

from idfworker.pull import consume_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handler(message):
    data = json.loads(message["Body"])
    logger.info(f"Received message: {data}")
    logger.info(f"Message ID: {message['MessageId']}")
    logger.info(f"Receipt handle: {message['ReceiptHandle']}")
    return 1


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
@click.option(
    "--dry_run",
    default=False,
    help="Dry run.",
)
def run(
    queue,
    experiment,
    batch_id,
    num_messages_to_process,
    visibility_timeout,
    wait_time,
    num_msgs_per_request,
    dry_run,
):
    """
    Run the worker.

    Args:
        queue (str, default="ml-for-bem"): The name of the SQS queue.
        experiment (str, default="idf/batch/test"): The name of the experiment.
        batch_id (str, default="test"): The Batch ID to filter messages.
        num_messages_to_process (int, default=10): The number of messages to process.
        visibility_timeout (int, default=120): The visibility timeout in seconds.
        wait_time (int, default=1): The time in seconds to wait for messages to arrive.
        num_msgs_per_request (int, default=10): The number of messages to receive per queue request.
        dry_run (bool, default=False): Dry run.
    """
    msg_handler = handler if not dry_run else lambda msg: logger.info(msg["Body"])
    consume_messages(
        queue=queue,
        experiment=experiment,
        batch_id=batch_id,
        num_messages_to_process=num_messages_to_process,
        num_msgs_per_request=num_msgs_per_request,
        visibility_timeout=visibility_timeout,
        wait_time=wait_time,
        handler=msg_handler,
    )


if __name__ == "__main__":
    run()
