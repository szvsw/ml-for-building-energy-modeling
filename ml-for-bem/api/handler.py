import runpod
import json
import numpy as np
import pandas as pd

from lightning.pytorch import Trainer

from ml.surrogate import Surrogate
from ml.predict import predict_ubem

surrogate = Surrogate.load_from_checkpoint("data/model.ckpt")
trainer = Trainer(
    accelerator="auto",
    strategy="auto",
    devices="auto",
)

# to serve locally:
# python ml/handler.py --rp_serve_api


def handler(job):
    job_input = job["input"]  # Access the input from the request.
    features = pd.DataFrame.from_dict(job_input["features"], orient="tight")
    schedules = np.array(job_input["schedules"])
    climate = np.array(job_input["climate"])
    sb_results, monthly_results, annual_results = predict_ubem(
        trainer,
        surrogate,
        features,
        schedules,
        climate,
        apply_cops=True,
        batch_size=32,
    )

    # convert column index level -1 to string
    sb_results.columns = sb_results.columns.set_levels(
        sb_results.columns.levels[-1].astype(str), level=-1
    )
    monthly_results.columns = monthly_results.columns.set_levels(
        monthly_results.columns.levels[-1].astype(str), level=-1
    )

    response = {
        "shoeboxes": sb_results.to_dict(orient="tight"),
        "monthly": monthly_results.to_dict(orient="tight"),
        "annual": annual_results.to_dict(orient="tight"),
    }
    return response


runpod.serverless.start({"handler": handler})
