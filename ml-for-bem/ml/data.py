from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.notebook import tqdm
from typing import Literal, Union
import lightning.pytorch as pl

from utils.nrel_uitls import CLIMATEZONES, CLIMATEZONES_LIST
from shoeboxer.builder import schedules_from_seed

# TODO: add a transform for weather data


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
            self.max_val = nn.parameter.Parameter(
                torch.tensor(
                    targets.max().values,
                    dtype=torch.float32,
                )
            )
            self.min_val = nn.parameter.Parameter(
                torch.tensor(
                    targets.min().values,
                    dtype=torch.float32,
                )
            )
        elif self.mode == "columnwise":
            self.max_val = nn.parameter.Parameter(
                torch.tensor(
                    targets.max(axis=0).values,
                    dtype=torch.float32,
                ).reshape(1, -1),
            )
            self.min_val = nn.parameter.Parameter(
                torch.tensor(
                    targets.min(axis=0).values,
                    dtype=torch.float32,
                ).reshape(1, -1),
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


class WeatherStdNormalTransform(nn.Module):
    def __init__(self, climate_array: np.ndarray):
        super().__init__()
        self.means = nn.parameter.Parameter(
            torch.tensor(
                climate_array,
                dtype=torch.float32,
            )
            .mean(dim=[0, 2])
            .reshape(1, -1, 1),
        )
        self.stds = nn.parameter.Parameter(
            torch.tensor(climate_array, dtype=torch.float32)
            .std(dim=[0, 2])
            .reshape(1, -1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.means) / self.stds


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
        # cz = self.climate_zones.iloc[index]
        epw_ix = self.epw_ixs.iloc[index]
        climate_data = self.climate_array[epw_ix]
        targets = self.targets[index]
        return self.features.iloc[index].values, schedules, climate_data, targets


class BuildingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        climate_array_path: str = "path/to/climate_array.npy",
        batch_size: int = 32,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_data_path = Path(data_dir) / "train" / "monthly.hdf"
        self.test_data_dir = Path(data_dir) / "test" / "monthly.hdf"
        self.space_config_path = Path(data_dir) / "train" / "space_definition.json"
        self.climate_array_path = climate_array_path
        self.batch_size = batch_size

    # TODO: implement data fetching from aws via prepare data
    # def prepare_data(self) -> None:
    #     return super().prepare_data()

    def setup(self, stage: str):
        with open(self.space_config_path, "r") as f:
            space_config = json.load(f)

        climate_array = np.load(self.climate_array_path)
        weather_transform = WeatherStdNormalTransform(climate_array)
        climate_array = (
            weather_transform(
                torch.tensor(
                    climate_array,
                    dtype=torch.float32,
                    device=next(weather_transform.parameters()).device,
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
        seen_epw_buiding_dataset = BuildingDataset(
            space_config,
            climate_array,
            self.train_data_path,
            key="batch_results",
        )

        target_transform = seen_epw_buiding_dataset.fit_target_transform(
            mode="columnwise"
        )

        self.seen_epw_train, self.seen_epw_val = random_split(
            seen_epw_buiding_dataset,
            [0.9, 0.1],
            generator=torch.Generator().manual_seed(42),
        )
        # building_data_unseen = BuildingDataset(
        #     space_config,
        #     climate_array,
        #     self.test_data_dir,
        #     key="batch_results",
        # )
        # building_data_unseen.load_target_transform(target_transform)

    def train_dataloader(self):
        return DataLoader(self.seen_epw_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return {"seen": DataLoader(self.seen_epw_val, batch_size=self.batch_size)}

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)


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

    weather_transform = WeatherStdNormalTransform(climate_array)
    climate_array = (
        weather_transform(
            torch.tensor(
                climate_array,
                dtype=torch.float32,
                device=next(weather_transform.parameters()).device,
            )
        )
        .detach()
        .cpu()
        .numpy()
    )

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
    for features, schedules, climate_data, targets in tqdm(dataloader):
        # print(features.shape, schedules.shape)
        features: torch.Tensor = features.float().to("cuda")
        schedules: torch.Tensor = schedules.float().to("cuda")
        climate_data: torch.Tensor = climate_data.float().to("cuda")
        targets: torch.Tensor = targets.float().to("cuda")
        timeseries: torch.Tensor = torch.cat([climate_data, schedules], dim=1)
        i += 1
        if i > 100:
            break
    e = time.time()
    print(f"{i} batches in {e-s:0.2f} seconds, {(e-s)/i:0.3f} seconds per batch")
    data.features.head()
