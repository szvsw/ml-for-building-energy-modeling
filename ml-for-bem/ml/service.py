from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

from bentoml.io import Multipart, NumpyNdarray, PandasDataFrame

from lightning.pytorch import Trainer
from ml.surrogate import Surrogate
from ml.predict import predict_ubem

ubem_runner = bentoml.pytorch_lightning.get("ubem:latest").to_runner()

svc = bentoml.Service("ubem", runners=[ubem_runner])

@svc.api(
    input=Multipart(
        features=PandasDataFrame(
            orient="tight",
            shape=(-1,50),
            dtype=np.float32,
            enforce_shape=False,
            enforce_dtype=False,
        ),
        schedules=NumpyNdarray(
            shape=(-1, 3,8760),
            dtype=np.float32,
            enforce_dtype=True,
            enforce_shape=True,
        ),
        climate=NumpyNdarray(
            shape=(-1, 7,8760),
            dtype=np.float32,
            enforce_dtype=True,
            enforce_shape=True,
        ),
    ),
    output=PandasDataFrame(
        orient="tight",
        shape=(-1,48),
        dtype=np.float32,
        enforce_dtype=True,
        enforce_shape=True,
    ),
)
def predict(features: pd.DataFrame, schedules: np.ndarray, climate: np.ndarray) -> pd.DataFrame:
    trainer = Trainer(accelerator="auto",devices="auto",strategy="auto")
    surrogate = None # TODO: load from runner?
    shoebox_predictions, monthly_predictions, annual_predictions = predict_ubem(
        trainer=trainer,
        surrogate=surrogate,
        features=features,
        schedules=schedules,
        climate=climate,
        apply_cops=False,
        batch_size=32,
    )
    return monthly_predictions
