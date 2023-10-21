from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from typing import Literal, Union

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


class MinMaxTransform(nn.Module):
    def __init__(self, targets: pd.DataFrame, mode: Literal["columnwise", "global"]):
        super().__init__()
        self.mode = mode
        if self.mode == "global":
            self.max_val = nn.parameter.Parameter(torch.tensor(targets.max().values))
            self.min_val = nn.parameter.Parameter(torch.tensor(targets.min().values))
        elif self.mode == "columnwise":
            self.max_val = nn.parameter.Parameter(
                torch.tensor(targets.max(axis=0).values).reshape(1, -1)
            )
            self.min_val = nn.parameter.Parameter(
                torch.tensor(targets.min(axis=0).values).reshape(1, -1)
            )
        self.columns = targets.columns

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, x: torch.Tensor, as_df: bool = False) -> torch.Tensor:
        vals = x * (self.max_val - self.min_val) + self.min_val
        if as_df:
            return pd.DataFrame(vals.detach().cpu().numpy(), columns=self.columns)
        else:
            return vals


class BuildingDataset(Dataset):
    def __init__(
        self,
        space_config,
        climate_array,
        path=None,
        key="batch_results",
    ):
        df = pd.read_hdf(path, key=key)
        features = df.index.to_frame(index=False)
        targets = df.reset_index(drop=True)
        mask = ~features["error"]
        features = features[mask]
        targets = targets[mask]
        self.targets_untransformed = targets
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

    def fit_target_transform(self, mode: Literal["columnwise", "global"]):
        self.target_transform = MinMaxTransform(
            self.targets_untransformed, mode=mode
        ).cpu()
        with torch.no_grad():
            self.targets = (
                self.target_transform(
                    torch.tensor(
                        self.targets_untransformed.values,
                        device=next(self.target_transform.parameters()).device,
                        dtype=torch.float32,
                    )
                )
                .cpu()
                .numpy()
            )
        return self.target_transform

    def load_target_transform(self, transform: nn.Module):
        self.target_transform = transform
        with torch.no_grad():
            self.targets = (
                self.target_transform(
                    torch.tensor(
                        self.targets_untransformed.values,
                        device=next(transform.parameters()).device,
                        dtype=torch.float32,
                    )
                )
                .cpu()
                .numpy()
            )
        return self.target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        schedule_seed = self.schedules_seed.iloc[index]
        schedules = schedules_from_seed(schedule_seed)
        cz = self.climate_zones.iloc[index]
        epw_ix = self.epw_ixs.iloc[index]
        climate_data = self.climate_array[epw_ix]
        targets = self.targets[index]
        return self.features.iloc[index].values, schedules, climate_data, cz, targets


if __name__ == "__main__":
    from pathlib import Path
    import time
    import json
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    with open("data/space_definition.json", "r") as f:
        space_config = json.load(f)

    climate_array = np.load(Path("data") / "epws" / "global_climate_array.npy")

    data = BuildingDataset(
        space_config,
        climate_array,
        "data/hdf5/full_climate_zone/v3/train/monthly.hdf",
        key="batch_results",
    )
    target_transform = data.fit_target_transform(mode="columnwise")

    # make a dataloader
    dataloader = DataLoader(data, batch_size=128, shuffle=True)

    i = 0
    s = time.time()
    for features, schedules, climate_data, cz, targets in tqdm(dataloader):
        # print(features.shape, schedules.shape)
        features: torch.Tensor = features.float().to("cuda")
        schedules: torch.Tensor = schedules.float().to("cuda")
        climate_data: torch.Tensor = climate_data.float().to("cuda")
        targets: torch.Tensor = targets.float().to("cuda")
        timeseries: torch.Tensor = torch.concat([climate_data, schedules], dim=1)
        i += 1
    e = time.time()
    print(f"{i} batches in {e-s:0.2f} seconds, {(e-s)/i:0.3f} seconds per batch")
    data.features.head()
