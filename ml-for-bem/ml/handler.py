import runpod
import numpy as np
import pandas as pd

from lightning.pytorch import Trainer

from ml.surrogate import Surrogate
from ml.predict import predict_ubem

surrogate = Surrogate.load_from_checkpoint("data/model-with-transform-configs.ckpt")
trainer = Trainer(
    accelerator="auto",
    strategy="auto",
    devices="auto",
)


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

    return sb_results.to_dict(orient="tight")


runpod.serverless.start({"handler": handler})
