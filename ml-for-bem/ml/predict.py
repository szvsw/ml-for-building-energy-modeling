from pathlib import Path
import pandas as pd
import numpy as np
import wandb
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from ml.train import Surrogate
from ml.data import PredictBuildingDataset


def fetch_model(
    registry="ml-for-building-energy-modeling/model-registry",
    model: str = "Global UBEM Shoebox Surrogate with Combined TS Embedder",
    tag: str = "latest",
    resource: str = "model.ckpt",
) -> Surrogate:
    """
    Fetches a surrogate model from the W&B cloud.

    Args:
        registry (str): The W&B registry to fetch the model from.
        model (str): The model name.
        tag (str): The model tag.
        resource (str): The file resource to fetch from within the model artifact.

    Returns:
        surrogate (Surrogate): The surrogate model.
    """
    api = wandb.Api()
    local_dir = Path("data") / "models" / tag
    model_str = f"{registry}/{model}:{tag}"
    surrogate_artifact = api.artifact(model_str, type="model")
    pth = surrogate_artifact.get_path(resource)
    model_path = pth.download(local_dir)
    surrogate = Surrogate.load_from_checkpoint(model_path)
    return surrogate


# TODO: Make sure that climate array is transformed
def predict_ubem(
    trainer: Trainer,
    surrogate: Surrogate,
    features: pd.DataFrame,
    schedules: np.ndarray,
    climate: np.ndarray,
    batch_size=32 * 32,
):
    """
    Predicts the energy consumption of an UBEM dataframe.  Assumes a single epw weather array.

    Args:
        trainer (Trainer): The lightning trainer; can be configured to handle various GPU strategies etc.
        surrogate (Surrogate): The surrogate model to use for prediction.
        features (pd.DataFrame): The UBEM dataframe (untransformed); template_idx column used in dataloader to match schedules
        schedules (np.ndarray): The schedules array (n_templates, n_schedules, 8760)
        climate (np.ndarray): The climate array (n_weather_timeseries, 8760)

    Returns:
        predictions (pd.DataFrame): The predicted energy consumption (untransformed), kWh/m2 per month for perim/core/heating/cooling per shoebox
    """
    space_config = surrogate.space_config
    dataset = PredictBuildingDataset(features, schedules, climate, space_config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = trainer.predict(surrogate, dataloader)
    predictions = pd.DataFrame(
        predictions.cpu().numpy(),
        columns=surrogate.target_transform.columns,
    )
    predictions = predictions.set_index(features.index)
    return predictions
