from pathlib import Path
import boto3
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Literal, Union
import lightning.pytorch as pl

from utils.nrel_uitls import CLIMATEZONES, CLIMATEZONES_LIST
from shoeboxer.builder import schedules_from_seed

# TODO: store and fetch weather data from s3
# TODO: store and return the weather transform


def transform_dataframe(space_config, features):
    """
    Transforms a dataframe of features into a dataframe of features that can be used as input to a neural network

    Args:
        space_config (dict): a dictionary defining the space of the features.  Should be formatted as:
            {
                "feature_name": {
                    "mode": "Continuous" or "Onehot",
                    "min": float (only for Continuous features),
                    "max": float (only for Continuous features),
                    "option_count": int (only for Onehot features),
                }
            }
        features (pd.DataFrame): a dataframe of features

    Returns:
        df: a dataframe of features that can be used as input to a neural network
    """

    # Create the dataframe
    df = pd.DataFrame()

    # check that the space_config and features are compatible
    for key in space_config:
        assert (
            key in features.columns
        ), f"{key} was in the provided space definition but not in the features dataframe"
    for column in features.columns:
        assert (
            column in space_config
        ), f"{column} was in the features dataframe but not in the provided space definition"

    # Do the conversion
    for key in space_config:
        # Get the current data
        column_data = features[key]

        if space_config[key]["mode"] == "Continuous":
            # Convert continuous parameters
            min_val = space_config[key]["min"]
            max_val = space_config[key]["max"]
            column_data = (column_data - min_val) / (max_val - min_val)

            # Store the converted data
            df[key] = column_data

        elif space_config[key]["mode"] == "Onehot":
            # Convert onehot parameters
            onehots = np.zeros((len(column_data), space_config[key]["option_count"]))
            column_data = column_data.astype(int)
            onehots[np.arange(len(column_data)), column_data] = 1
            column_data = onehots

            # Store the converted data
            for i in range(space_config[key]["option_count"]):
                df[f"{key}_{i}"] = column_data[:, i]

        else:
            # throw error for other modes
            raise ValueError(
                f"Unknown mode {space_config[key]['mode']} for feature: {key}"
            )
    return df


class MinMaxTransform(nn.Module):
    """
    Transforms a tensor of values to the range [0, 1] by subtracting the minimum value and dividing by the range
    """

    def __init__(
        self,
        targets: pd.DataFrame,
        mode: Literal["columnwise", "global"] = "columnwise",
    ):
        """
        Create a MinMaxTransform which can be used to transform tensors to the range [0, 1]

        Args:
            targets (pd.DataFrame): a dataframe of targets
            mode (str): either "columnwise" or "global".  If "columnwise", the maximum and minimum values are computed for each column.  If "global", the maximum and minimum values are computed for the entire dataframe.

        Returns:
            MinMaxTransform: a transform which can be used to transform tensors to the range [0, 1]
        """

        super().__init__()

        self.mode = mode

        if self.mode == "global":
            # Use the max of the entire dataframe
            self.max_val = nn.parameter.Parameter(
                torch.tensor(
                    targets.max().values,
                    dtype=torch.float32,
                )
            )
            # Use the min of the entire dataframe
            self.min_val = nn.parameter.Parameter(
                torch.tensor(
                    targets.min().values,
                    dtype=torch.float32,
                )
            )
        elif self.mode == "columnwise":
            # use the max of each column
            self.max_val = nn.parameter.Parameter(
                torch.tensor(
                    targets.max(axis=0).values,
                    dtype=torch.float32,
                ).reshape(1, -1),
            )
            # use the min of each column
            self.min_val = nn.parameter.Parameter(
                torch.tensor(
                    targets.min(axis=0).values,
                    dtype=torch.float32,
                ).reshape(1, -1),
            )

        # Store the columnar labels for future use
        self.columns = targets.columns

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform a tensor to the range [0, 1]

        Args:
            x (torch.Tensor): a tensor of values

        Returns:
            x_transformed (torch.Tensor): a tensor of values transformed to the range [0, 1]
        """

        x_transformed = (x - self.min_val) / (self.max_val - self.min_val)

        return x_transformed

    def inverse_transform(self, x: torch.Tensor, as_df: bool = False) -> torch.Tensor:
        """
        Transform a tensor from the range [0, 1] to the original range

        Args:
            x (torch.Tensor): a tensor of values in the range [0, 1]
            as_df (bool, default False): if True, return a dataframe using the original column names.  Otherwise, return a tensor

        Returns:
            x_untransformed (torch.Tensor): a tensor of values transformed from the range [0, 1] to the original range
        """

        vals = x * (self.max_val - self.min_val) + self.min_val

        if as_df:
            return pd.DataFrame(vals.detach().cpu().numpy(), columns=self.columns)

        else:
            return vals


class StdNormalTransform(nn.Module):
    """
    Transforms a tensor of values to the standard normal distribution by subtracting the mean and dividing by the standard deviation
    """

    def __init__(self, targets: pd.DataFrame):
        """
        Create a StdNormalTransform which can be used to transform tensors to the standard normal distribution

        Args:
            targets (pd.DataFrame): a dataframe of targets

        Returns:
            StdNormalTransform: a transform which can be used to transform tensors to the standard normal distribution
        """
        super().__init__()
        self.std = nn.parameter.Parameter(
            torch.tensor(
                targets.std(axis=0).values,
                dtype=torch.float32,
            ).reshape(1, -1),
        )
        self.mean = nn.parameter.Parameter(
            torch.tensor(
                targets.mean(axis=0).values,
                dtype=torch.float32,
            ).reshape(1, -1),
        )

        # Get the column names for future use
        self.columns = targets.columns

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform a tensor to the standard normal distribution

        Args:
            x (torch.Tensor): a tensor of values

        Returns:
            x_transformed (torch.Tensor): a tensor of values transformed to the standard normal distribution
        """

        x_transformed = (x - self.mean) / self.std

        return x_transformed

    def inverse_transform(self, x: torch.Tensor, as_df: bool = False) -> torch.Tensor:
        """
        Transform a tensor from the standard normal distribution to the original distribution

        Args:
            x (torch.Tensor): a tensor of values in the standard normal distribution
            as_df (bool, default False): if True, return a dataframe using the original column names.  Otherwise, return a tensor

        Returns:
            x_untransformed (torch.Tensor): a tensor of values transformed from the standard normal distribution to the original distribution
        """

        vals = x * self.std + self.mean

        if as_df:
            return pd.DataFrame(vals.detach().cpu().numpy(), columns=self.columns)

        else:
            return vals


class WeatherStdNormalTransform(nn.Module):
    """
    Transforms a tensor of weather data to the standard normal distribution by subtracting the mean and dividing by the standard deviation
    """

    def __init__(self, climate_array: np.ndarray):
        """
        Create a WeatherStdNormalTransform which can be used to transform tensors of weather data to the standard normal distribution
        Each weather channel is transformed independently

        Args:
            climate_array (np.ndarray): an array of climate data.  Assumed shape is (n_epws, n_weather_channels, n_timesteps)

        Returns:
            WeatherStdNormalTransform: a transform which can be used to transform tensors of weather data to the standard normal distribution
        """
        super().__init__()

        # Compute the mean and standard deviation of each weather channel
        # To do so, we take the mean and standard deviation over the epws and timesteps

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
        """
        Transform a tensor of weather data to the standard normal distribution

        Args:
            x (torch.Tensor): a tensor of weather data.  Shape is (batch_size, n_weather_channels, n_timesteps)

        Returns:
            x_transformed (torch.Tensor): a tensor of weather data transformed to the standard normal distribution
        """

        x_transformed = (x - self.means) / self.stds

        return x_transformed


class BuildingDataset(Dataset):
    """
    A dataset of building features and targets
    """

    def __init__(
        self,
        space_config,
        climate_array,
        path=None,
        key="batch_results",
    ):
        """
        Create a BuildingDataset

        Args:
            space_config (dict): a dictionary defining the space of the features.  Should be formatted as:
                {
                    "feature_name": {
                        "mode": "Continuous" or "Onehot",
                        "min": float (only for Continuous features),
                        "max": float (only for Continuous features),
                        "option_count": int (only for Onehot features),
                    }
                }
            climate_array (np.ndarray): an array of climate data.  Assumed shape is (n_epws, n_weather_channels, n_timesteps)
            path (str, default None): the path to the hdf5 file containing the data
            key (str, default "batch_results"): the key in the hdf5 file containing the data

        Returns:
            BuildingDataset: a dataset of building features and targets
        """

        # Load the dataset
        df = pd.read_hdf(path, key=key)

        # Extract the features which are the index
        features = df.index.to_frame(index=False)

        # Extract the targets
        targets = df.reset_index(drop=True)

        # Drop errored rows
        mask = ~features["error"]
        features = features[mask]
        targets = targets[mask]

        # Store the targets in their original form
        self.targets_untransformed = targets

        # Store the climate zones
        self.climate_zones = features["climate_zone"].apply(
            lambda x: CLIMATEZONES_LIST[int(x)]
        )
        self.cz_ixs = features["climate_zone"].astype(int)

        # Store the epw indices
        self.epw_ixs = features["base_epw"].astype(int)

        # Store the schedules seed
        self.schedules_seed = features["schedules_seed"].astype(int)

        # Drop unnecessary columns, leaving only features
        features = features.drop("id", axis=1)
        features = features.drop("error", axis=1)
        features = features.drop("climate_zone", axis=1)
        features = features.drop("base_epw", axis=1)
        features = features.drop("schedules_seed", axis=1)

        # Store the original features
        self.features_untransformed = features

        # Transform the features and convert to a f32 precision
        self.features = transform_dataframe(space_config, features)
        self.features = self.features.astype(np.float32)

        # store the climate array and space config
        self.space_config = space_config
        self.climate_array = climate_array

    def fit_target_transform(self):
        """
        Fit a transform to the targets and transform the targets.
        Should only be used on the training dataset's targets

        Returns:
            transform (nn.Module): a transform which can be used to transform targets

        """

        self.target_transform = MinMaxTransform(self.targets_untransformed).cpu()
        # self.target_transform = StdNormalTransform(self.targets_untransformed).cpu()

        # Apply the transform and store as  numpy
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
        """
        Load and store a transform for the targets and transform the targets
        Should be used on the validation and test datasets' targets

        Args:
            transform (nn.Module): a transform which can be used to transform targets

        Returns:
            transform (nn.Module): the same transform which can be used to transform targets
        """

        # Store the transform
        self.target_transform = transform

        # apply the transform and save as numpy
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
        """
        Get a single sample from the dataset.  Handles fetching the climate data and constructing schedules from seed

        Args:
            index (int): the index of the sample

        Returns:
            features (np.ndarray): the features
            schedules (np.ndarray): the schedules
            climate_data (np.ndarray): the climate data
            targets (np.ndarray): the targets
        """

        # get the features, which are already transformed
        features = self.features.iloc[index].values

        # Get the schedule seed and make the schedule as f32 precision
        schedule_seed = self.schedules_seed.iloc[index]
        schedules = schedules_from_seed(schedule_seed).astype(np.float32)

        # get the epw index and load the climate data
        epw_ix = self.epw_ixs.iloc[index]
        climate_data = self.climate_array[epw_ix]
        # cz = self.climate_zones.iloc[index]

        # get the targets, which are already transformed
        targets = self.targets[index]

        return features, schedules, climate_data, targets


class BuildingDataModule(pl.LightningDataModule):
    """
    A LightningDataModule for the BuildingDataset, which abstracts loading
    and preparing the data for training, validation, and testing
    """

    def __init__(
        self,
        bucket: str = "ml-for-bem",
        remote_experiment: str = "full_climate_zone/v3",
        data_dir: str = "path/to/dir",
        climate_array_path: str = "path/to/climate_array.npy",
        batch_size: int = 32,
    ):
        """
        Create a BuildingDataModule which can be used to load and prepare the data for training, validation, and testing.
        The data is stored on s3 and downloaded to the local data_dir if not there already

        Args:
            bucket: the name of the s3 bucket where the data is stored
            remote_experiment: the name of the experiment on s3
            data_dir: the local directory where the data should be stored
            climate_array_path: the path to the climate array on the local filesystem
            batch_size (int): the batch size to use for training; validation and testing batch sizes are 32x this value

        Returns:
            BuildingDataModule: a LightningDataModule which can be used to load and prepare the data for training, validation, and testing

        """
        super().__init__()

        # Store the arguments
        self.data_dir = data_dir
        self.bucket = bucket
        self.batch_size = batch_size
        self.remote_experiment = remote_experiment

        # Make the paths
        self.experiment_root = Path(data_dir) / remote_experiment
        self.train_data_path = Path(self.experiment_root) / "train" / "monthly.hdf"
        self.test_data_dir = Path(self.experiment_root) / "test" / "monthly.hdf"
        self.space_config_path = (
            Path(self.experiment_root) / "train" / "space_definition.json"
        )
        self.climate_array_path = climate_array_path

    def prepare_data(self):
        # TODO: download global_climate_array.npy

        # Download the data from s3 if it doesn't exist locally
        s3 = boto3.client("s3")
        if not os.path.exists(self.train_data_path):
            os.makedirs(self.train_data_path.parent, exist_ok=True)
            s3.download_file(
                self.bucket,
                f"{self.remote_experiment}/train/monthly.hdf",
                self.train_data_path,
            )
        if not os.path.exists(self.test_data_dir):
            os.makedirs(self.test_data_dir.parent, exist_ok=True)
            s3.download_file(
                self.bucket,
                f"{self.remote_experiment}/test/monthly.hdf",
                self.test_data_dir,
            )
        if not os.path.exists(self.space_config_path):
            os.makedirs(self.space_config_path.parent, exist_ok=True)
            s3.download_file(
                self.bucket,
                f"{self.remote_experiment}/train/space_definition.json",
                self.space_config_path,
            )

    def setup(self, stage: str):
        # Load the data, apply transforms, make datasets

        # Load the space config definition
        with open(self.space_config_path, "r") as f:
            space_config = json.load(f)

        # Load the climate array and apply weather transform
        climate_array = np.load(self.climate_array_path)
        weather_transform = WeatherStdNormalTransform(climate_array)
        self.climate_array = (
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
        self.climate_array = self.climate_array.astype(np.float32)
        self.weather_transform = weather_transform

        # Make the dataset for seen weather data and fit a target transform
        seen_epw_buiding_dataset = BuildingDataset(
            space_config,
            self.climate_array,
            self.train_data_path,
            key="batch_results",
        )
        target_transform = seen_epw_buiding_dataset.fit_target_transform()
        self.target_transform = target_transform

        # Split the dataset for seen weather data into training and validation
        self.seen_epw_training_set, self.seen_epw_validation_set = random_split(
            seen_epw_buiding_dataset,
            [0.9, 0.1],
            generator=torch.Generator().manual_seed(42),
        )

        # Make the dataset for unseen weather data and load the target transform
        unseen_epw_validation_set = BuildingDataset(
            space_config,
            self.climate_array,
            self.test_data_dir,
            key="batch_results",
        )
        unseen_epw_validation_set.load_target_transform(target_transform)
        self.unseen_epw_validation_set = unseen_epw_validation_set

    def train_dataloader(self):
        return DataLoader(self.seen_epw_training_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return [
            DataLoader(self.seen_epw_validation_set, batch_size=self.batch_size * 32),
            DataLoader(self.unseen_epw_validation_set, batch_size=self.batch_size * 32),
        ]

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)


if __name__ == "__main__":
    """
    Debugging/testing dataloaders
    """

    import time
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
    target_transform = data.fit_target_transform()

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
