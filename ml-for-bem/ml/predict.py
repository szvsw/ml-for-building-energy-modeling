import pandas as pd
import numpy as np
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from ml.train import Surrogate
from ml.data import PredictBuildingDataset


# TODO: Make sure that climate array is transformed
def predict(
    trainer: Trainer,
    model: Surrogate,
    features: pd.DataFrame,
    schedules: np.ndarray,
    climate: np.ndarray,
    batch_size=32 * 32,
):
    dataset = PredictBuildingDataset(features, schedules, climate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = trainer.predict(model, dataloader)
    predictions = pd.DataFrame(
        predictions.cpu().numpy(),
        columns=model.target_transform.columns,
    )
    predictions = predictions.set_index(features.index)
    return predictions
