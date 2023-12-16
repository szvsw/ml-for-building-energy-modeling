import os

import base64
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from archetypal import settings

from shoeboxer.batch import batch_sim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Lambda module loaded.")

# Check if we are running on Windows or Linux using os
if os.name == "nt":
    settings.ep_version == "22.2.0"
    settings.energyplus_location = Path("C:/EnergyPlusV22-2-0")
else:
    settings.ep_version == "22.2.0"
    settings.energyplus_location = Path("/usr/local/EnergyPlus-22-2-0")


def handler(event, context):
    """
    event: dict
    context: lambda_context
    """

    body = event["body"]
    if type(body) == str:
        # print(body)
        body = json.loads(body)
    with open("/tmp/epw.epw", "wb") as f:
        f.write(base64.b64decode(body["climate"]))

    # get the parameters from the event
    features = body["features"]
    timeseries = body["schedules"]
    is_cloud_lambda = body["is_cloud_lambda"]
    climate = "/tmp/epw.epw"
    logger.info(f"features: {len(features)}")

    features = pd.DataFrame.from_dict(features, orient="tight")
    timeseries = np.array(timeseries)

    # # load the climate file

    # # run the simulation
    monthly_results = batch_sim(
        features=features,
        timeseries=timeseries,
        climate=climate,
        folder="/tmp" if is_cloud_lambda else None,
        # parallel=2,
    )
    logger.info(f"monthly_dfs: {len(monthly_results)}")
    monthly_results.columns = monthly_results.columns.set_levels(
        monthly_results.columns.levels[-1].astype(str), level=-1
    )
    return monthly_results.to_dict(orient="tight")
    return {"message": "success"}
