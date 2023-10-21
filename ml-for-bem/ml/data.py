import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

from utils.nrel_uitls import CLIMATEZONES, CLIMATEZONES_LIST
from shoeboxer.builder import schedules_from_seed


def transform_dataframe(space_config, features):
    df = pd.DataFrame()
    for key in space_config:
        assert (
            key in features.columns
        ), f"{key} was in the provided space definition but not in the features dataframe"
    for column in features.columns:
        assert (
            column in space_config
        ), f"{column} was in the features dataframe but not in the provided space definition"

    for key in space_config:
        column_data = features[key]
        if space_config[key]["mode"] == "Continuous":
            min_val = space_config[key]["min"]
            max_val = space_config[key]["max"]
            column_data = (column_data - min_val) / (max_val - min_val)
            df[key] = column_data
        elif space_config[key]["mode"] == "Onehot":
            onehots = np.zeros((len(column_data), space_config[key]["option_count"]))
            column_data = column_data.astype(int)
            onehots[np.arange(len(column_data)), column_data] = 1
            column_data = onehots
            for i in range(space_config[key]["option_count"]):
                df[f"{key}_{i}"] = column_data[:, i]
    return df


class BuildingDataset(Dataset):
    def __init__(
        self,
        space_config,
        climate_array,
        path,
        key="batch_results",
    ):
        df = pd.read_hdf(path, key=key)
        features = df.index.to_frame(index=False)
        targets = df.reset_index(drop=True)
        mask = ~features["error"]
        features = features[mask]
        targets = targets[mask]
        self.climate_zones = features["climate_zone"].apply(
            lambda x: CLIMATEZONES_LIST[int(x)]
        )
        self.cz_ixs = features["climate_zone"].astype(int)
        self.epw_ixs = features["base_epw"].astype(int)
        self.schedules_seed = features["schedules_seed"].astype(int)

        features = features.drop("id", axis=1)
        features = features.drop("error", axis=1)
        features = features.drop("climate_zone", axis=1)
        features = features.drop("base_epw", axis=1)
        features = features.drop("schedules_seed", axis=1)

        self.features_untransformed = features
        self.features = transform_dataframe(space_config, features)
        self.space_config = space_config
        self.climate_array = climate_array

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        schedule_seed = self.schedules_seed.iloc[index]
        schedules = schedules_from_seed(schedule_seed)
        cz = self.climate_zones.iloc[index]
        epw_ix = self.epw_ixs.iloc[index]
        climate_data = self.climate_array[epw_ix]
        return self.features.iloc[index].values, schedules, climate_data, cz
