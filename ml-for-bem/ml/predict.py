from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from ml.data import PredictBuildingDataset
from ml.surrogate import Surrogate


# TODO: Make sure that climate array is transformed
def predict_ubem(
    trainer: Trainer,
    surrogate: Surrogate,
    features: pd.DataFrame,
    schedules: np.ndarray,
    climate: np.ndarray,
    apply_cops: bool = True,
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
        apply_cops (bool): Whether to apply the COPs to the heating/cooling energy
        batch_size (int): The batch size to use for prediction

    Returns:
        predictions (pd.DataFrame): The predicted energy consumption (unweighted), kWh/m2 per month for perim/core/heating/cooling per shoebox
        weighted_predictions (pd.DataFrame): The predicted energy consumption (weighted), kWh/m2 per month for perim/core/heating/cooling per building
        annual_predictions (pd.DataFrame): The predicted energy consumption (weighted), kWh/m2 per year for perim/core/heating/cooling per building
    """
    space_config = surrogate.space_config
    dataset = PredictBuildingDataset(features, schedules, climate, space_config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    shoebox_predictions = trainer.predict(surrogate, dataloader)
    shoebox_predictions = torch.cat(shoebox_predictions)
    try:
        shoebox_predictions = shoebox_predictions.cpu().numpy()
    except:
        shoebox_predictions = shoebox_predictions.numpy()

    # TODO: consider doing aggregation in torch
    shoebox_predictions = pd.DataFrame(
        shoebox_predictions,
        columns=surrogate.target_transform.columns,
    )

    building_id = features.building_id
    # TODO: make sure the index order hasn't changed
    # TODO: maybe this index settting should actually be done on the original features index
    # and then we just copy over the multiindex
    shoebox_predictions.index = pd.MultiIndex.from_arrays([building_id, features.index])
    # weight each of the shoeboxes
    monthly_predictions = shoebox_predictions * features["weight"].values.reshape(-1, 1)

    # get the core/perim weight balance
    # TODO: this weight should be precomputed
    perim_depth = features.height * features.floor_2_facade
    core_depth = perim_depth * features.core_2_perim
    total_depth = perim_depth + core_depth
    core_weight = core_depth / total_depth
    perim_weight = perim_depth / total_depth

    # weight the core/perim predictions
    monthly_predictions["Core"] = monthly_predictions.Core * core_weight.values.reshape(
        -1, 1
    )
    monthly_predictions[
        "Perimeter"
    ] = monthly_predictions.Perimeter * perim_weight.values.reshape(-1, 1)

    # combine heating and cooling from core/perim
    # already applied the weight factors so summation is correct
    monthly_predictions = monthly_predictions.T.groupby(level=[1, 2]).sum().T

    if apply_cops:
        # apply COPs
        monthly_predictions["Heating"] = monthly_predictions["Heating"] / features[
            "cop_heating"
        ].values.reshape(-1, 1)
        monthly_predictions["Cooling"] = monthly_predictions["Cooling"] / features[
            "cop_cooling"
        ].values.reshape(-1, 1)

    # aggregate by building
    # already applied weight factors so summation is correct
    monthly_predictions = monthly_predictions.groupby("building_id").sum()

    # aggregate to annual
    annual_predictions = monthly_predictions.T.groupby(level=[0]).sum().T
    return shoebox_predictions, monthly_predictions, annual_predictions
