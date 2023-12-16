import json
import logging
import os
from pathlib import Path
from typing import List, Literal, Union

import boto3
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from shoeboxer.schedules import schedules_from_seed
from utils.nrel_uitls import CLIMATEZONES, CLIMATEZONES_LIST

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# TODO: store and fetch weather data from s3
# TODO: store and return the weather transform


def transform_dataframe(space_config, features, allow_oob=False):
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
        allow_oob (bool, default False): if True, allow values outside of the range [0, 1] for continuous features.  If False, throw an error if a value is outside of the range [0, 1]

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
    # for column in features.columns:
    #     assert (
    #         column in space_config
    #     ), f"{column} was in the features dataframe but not in the provided space definition"

    # Do the conversion
    for key in space_config:
        # Get the current data
        column_data_og = features[key]

        if space_config[key]["mode"] == "Continuous":
            # Convert continuous parameters
            min_val = space_config[key]["min"]
            max_val = space_config[key]["max"]
            column_data = (column_data_og - min_val) / (max_val - min_val)
            if not allow_oob:
                assert column_data.min() >= 0.0 and column_data.max() <= 1.0, (
                    f"Values for {key} are outside of the range "
                    + f"[{space_config[key]['min']}, {space_config[key]['max']}]!\n"
                    + f"min: {column_data_og.min()}, max: {column_data_og.max()}"
                )

            else:
                # Clip the values
                # warn if values are outside of the range
                if column_data.min() < 0.0 or column_data.max() > 1.0:
                    logger.warning(
                        f"Values for {key} are outside of the range "
                        + f"[{space_config[key]['min']}, {space_config[key]['max']}]! "
                        + f"Provided data min: {column_data_og.min()}, max: {column_data_og.max()}. "
                        + "Data will be clipped before prediction."
                    )
                column_data = column_data.clip(0.0, 1.0)

            # Store the converted data
            df[key] = column_data

        elif space_config[key]["mode"] == "Onehot":
            # Convert onehot parameters
            onehots = np.zeros((len(column_data), space_config[key]["option_count"]))
            column_data_og = column_data_og.astype(int)
            onehots[np.arange(len(column_data_og)), column_data_og] = 1
            column_data = onehots
            assert column_data_og.max() < space_config[key]["option_count"], (
                f"Max Value for {key} is {column_data_og.max()}, "
                f"but option count is {space_config[key]['option_count']}"
            )

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

    def __init__(self, climate_array: np.ndarray, channel_names: List[str]):
        """
        Create a WeatherStdNormalTransform which can be used to transform tensors of weather data to the standard normal distribution
        Each weather channel is transformed independently

        Args:
            climate_array (np.ndarray): an array of climate data.  Assumed shape is (n_epws, n_weather_channels, n_timesteps)
            channel_names (List[str]): a list of channel names.  Should be the same length as the number of weather channels

        Returns:
            WeatherStdNormalTransform: a transform which can be used to transform tensors of weather data to the standard normal distribution
        """
        super().__init__()

        # Compute the mean and standard deviation of each weather channel
        # To do so, we take the mean and standard deviation over the epws and timesteps

        self.channel_names = channel_names
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


# TODO: add a transform for weather data
class PredictBuildingDataset(Dataset):
    def __init__(
        self,
        features: pd.DataFrame,
        schedules: np.ndarray,
        climate_array: np.ndarray,
        space_config: dict,
    ):
        """
        Create a PredictBuildingDataset

        Args:
            features (pd.DataFrame): a dataframe of features, untransformed (i.e. physical units); one column should be "template_idx", which will be used to select schedules.
            schedules (np.ndarray): an array of schedules (n_templates, n_timesteps, n_channels)
            climate_array (np.ndarray): an array of climate data.  Assumed shape is (n_weather_channels, n_timesteps)
            space_config: a dictionary defining the space of the features.  Should be formatted as:
                {
                    "feature_name": {
                        "mode": "Continuous" or "Onehot",
                        "min": float (only for Continuous features),
                        "max": float (only for Continuous features),
                        "option_count": int (only for Onehot features),
                    }
                }

        Returns:
            PredictBuildingDataset: a dataset of building features and schedules
        """

        # Store the features
        self.template_idx = features["template_idx"].astype(int)
        self.features_untransformed = features.drop("template_idx", axis=1)
        self.features = transform_dataframe(
            space_config, self.features_untransformed, allow_oob=True
        ).astype(np.float32)

        # Store the schedules
        self.schedules = schedules.astype(np.float32)
        assert (
            self.schedules.min() >= 0.0 and self.schedules.max() <= 1.0
        ), "Some of the provided schedules are not in the range [0, 1]!"

        # Store the climate array
        self.climate_array = climate_array.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.  Handles fetching the climate data and schedules

        Args:
            index (int): the index of the sample

        Returns:
            features (np.ndarray): the features
            schedules (np.ndarray): the schedules
            climate_data (np.ndarray): the climate data, untransformed
        """

        # get the features, which are already transformed
        features = self.features.iloc[index].values

        # Get the schedule seed and make the schedule as f32 precision
        template_idx = self.template_idx.iloc[index]
        schedules = self.schedules[template_idx].astype(np.float32)

        # get the epw index and load the climate data
        climate_data = self.climate_array

        return features, schedules, climate_data


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
        target_thresh: int = 100,
    ):
        """
        Create a BuildingDataset.  If no climate data is provided, then only features and schedules will be delivered.
        If "schedules_seed" is found in the space_config, then the schedules will be considered categorical and included in
        the features matrix, otherwise it will be full timeseries generated from the seed.

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
            climate_array (np.ndarray): an array of climate data.  Assumed shape is (n_epws, n_weather_channels, n_timesteps), if none, ignored.
            path (str, default None): the path to the hdf5 file containing the data
            key (str, default "batch_results"): the key in the hdf5 file containing the data
            target_thresh (int, default 100): the threshold for targets.  If any target is above this value, the sample is dropped
        Returns:
            BuildingDataset: a dataset of building features and targets
        """

        # Load the dataset
        df = pd.read_hdf(path, key=key)

        # Extract the features which are the index
        features = df.index.to_frame(index=False)

        # Extract the targets
        targets = df.reset_index(drop=True)
        self.target_thresh = target_thresh
        thresh_mask = (targets < self.target_thresh).all(axis=1)

        # Drop errored rows
        mask = (~features["error"]) & (thresh_mask)
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

        # Drop unnecessary columns, leaving only features
        features = features.drop("id", axis=1)
        features = features.drop("error", axis=1)
        features = features.drop("climate_zone", axis=1)
        features = features.drop("base_epw", axis=1)
        # Store the schedules seed
        if "schedules_seed" not in space_config.keys():
            self.schedules_seed = features["schedules_seed"].astype(int)
            features = features.drop("schedules_seed", axis=1)
            self.schedules_from_seed = True
        else:
            self.schedules_from_seed = False

        # Store the original features
        self.features_untransformed = features

        # Transform the features and convert to a f32 precision
        logger.info("Transforming features dataset...")
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
        if self.schedules_from_seed:
            schedule_seed = self.schedules_seed.iloc[index]
            schedules = schedules_from_seed(schedule_seed).astype(np.float32)
        else:
            schedules = None

        if self.climate_array is not None:
            # get the epw index and load the climate data
            epw_ix = self.epw_ixs.iloc[index]
            climate_data = self.climate_array[epw_ix]
            # cz = self.climate_zones.iloc[index]
        else:
            climate_data = None

        # get the targets, which are already transformed
        targets = self.targets[index]

        if schedules is None and climate_data is None:
            return features, targets
        elif schedules is None:
            return features, climate_data, targets
        elif climate_data is None:
            return features, schedules, targets
        else:
            return features, schedules, climate_data, targets


class BuildingHourlyDataset(BuildingDataset):
    def __init__(
        self,
        space_config,
        climate_array,
        path=None,
        key="batch_results",
        target_thresh: int = 100,
    ):
        """
        Create a BuildingDataset.  If no climate data is provided, then only features and schedules will be delivered.
        If "schedules_seed" is found in the space_config, then the schedules will be considered categorical and included in
        the features matrix, otherwise it will be full timeseries generated from the seed.

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
            climate_array (np.ndarray): an array of climate data.  Assumed shape is (n_epws, n_weather_channels, n_timesteps), if none, ignored.
            path (str, default None): the path to the hdf5 file containing the data
            key (str, default "batch_results"): the key in the hdf5 file containing the data
            target_thresh (int, default 100): the threshold for targets.  If any target is above this value, the sample is dropped
        Returns:
            BuildingDataset: a dataset of building features and targets
        """

        # Load the dataset
        df = pd.read_hdf(path, key=key)

        # Extract the features which are the index
        features = df.index.to_frame(index=False)

        # Extract the targets
        targets = df.reset_index(drop=True)
        self.target_thresh = target_thresh
        thresh_mask = (targets < self.target_thresh).all(axis=1)

        # Drop errored rows
        mask = (~features["error"]) & (thresh_mask)
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

        # Drop unnecessary columns, leaving only features
        features = features.drop("id", axis=1)
        features = features.drop("error", axis=1)
        features = features.drop("climate_zone", axis=1)
        features = features.drop("base_epw", axis=1)
        # Store the schedules seed
        if "schedules_seed" not in space_config.keys():
            self.schedules_seed = features["schedules_seed"].astype(int)
            features = features.drop("schedules_seed", axis=1)
            self.schedules_from_seed = True
        else:
            self.schedules_from_seed = False

        # Store the original features
        self.features_untransformed = features

        # Transform the features and convert to a f32 precision
        logger.info("Transforming features dataset...")
        self.features = transform_dataframe(space_config, features)
        self.features = self.features.astype(np.float32)

        # store the climate array and space config
        self.space_config = space_config
        self.climate_array = climate_array

    def __len__(self):
        return len(self.features) * 8760

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
        building_index = index // 8760
        hour_index = index % 8760
        if hour_index < 3:
            hour_index = 3
        features = self.features.iloc[building_index].values

        # Get the schedule seed and make the schedule as f32 precision
        if self.schedules_from_seed:
            schedule_seed = self.schedules_seed.iloc[building_index]
            schedules = schedules_from_seed(schedule_seed).astype(np.float32)
            schedules = schedules[:, hour_index - 3 : hour_index + 1]
            schedules = schedules.flatten()
        else:
            schedules = None

        if self.climate_array is not None:
            # get the epw index and load the climate data
            epw_ix = self.epw_ixs.iloc[building_index]
            climate_data_annual: np.ndarray = self.climate_array[epw_ix]
            # grab the current hour as well as the previous three hours
            # repeat the first three hours for the first three hours of the year
            climate_data = climate_data_annual[:, hour_index - 3 : hour_index + 1]
            climate_data = climate_data.flatten()

            # cz = self.climate_zones.iloc[index]
        else:
            climate_data = None

        # get the targets, which are already transformed
        targets = self.targets[building_index]
        # get every nth hour from teh flat list of many targets
        targets = targets[hour_index::8760]

        if schedules is None and climate_data is None:
            return features, targets
        elif schedules is None:
            return np.concatenate([features, climate_data]), targets
        elif climate_data is None:
            return np.concatenate([features, schedules]), targets
        else:
            return np.concatenate([features, schedules, climate_data]), targets


# TODO: Shuffle the data?
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
        climate_experiment: str = "weather/v1",
        batch_size: int = 32,
        val_batch_mult: int = 16,
    ):
        """
        Create a BuildingDataModule which can be used to load and prepare the data for training, validation, and testing.
        The data is stored on s3 and downloaded to the local data_dir if not there already

        Args:
            bucket: the name of the s3 bucket where the data is stored
            remote_experiment: the name of the experiment on s3
            data_dir: the local directory where the data should be stored
            climate_experiment: the name of the experiment on s3 where the climate data is stored
            batch_size (int): the batch size to use for training
            val_batch_mult (int): the multiplier to use on the training batch size to determine validation/testing batch size

        Returns:
            BuildingDataModule: a LightningDataModule which can be used to load and prepare the data for training, validation, and testing

        """
        super().__init__()

        # Store the arguments
        self.data_dir = data_dir
        self.bucket = bucket
        self.batch_size = batch_size
        self.val_batch_mult = val_batch_mult
        self.remote_experiment = remote_experiment
        self.climate_experiment = climate_experiment

        # Make the paths
        self.experiment_root = Path(data_dir) / remote_experiment
        self.train_data_path = Path(self.experiment_root) / "train" / "monthly.hdf"
        self.test_data_dir = Path(self.experiment_root) / "test" / "monthly.hdf"
        self.space_config_path = (
            Path(self.experiment_root) / "train" / "space_definition.json"
        )
        self.climate_experiment_root = (
            (Path(data_dir) / climate_experiment)
            if self.climate_experiment is not None
            else None
        )
        self.climate_array_path = (
            (self.climate_experiment_root / "global_climate_array.npy")
            if self.climate_experiment is not None
            else None
        )
        self.climate_timeseries_names_path = (
            (self.climate_experiment_root / "timeseries.json")
            if self.climate_experiment is not None
            else None
        )

    def prepare_data(self):
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
        if self.climate_experiment is not None:
            if not os.path.exists(self.climate_array_path):
                os.makedirs(self.climate_array_path.parent, exist_ok=True)
                s3.download_file(
                    self.bucket,
                    f"{self.climate_experiment}/global_climate_array.npy",
                    self.climate_array_path,
                )

            if not os.path.exists(self.climate_timeseries_names_path):
                os.makedirs(self.climate_timeseries_names_path.parent, exist_ok=True)
                s3.download_file(
                    self.bucket,
                    f"{self.climate_experiment}/timeseries.json",
                    self.climate_timeseries_names_path,
                )

    def setup(self, stage: str):
        # Load the data, apply transforms, make datasets

        # Load the space config definition
        with open(self.space_config_path, "r") as f:
            space_config = json.load(f)
        self.space_config = space_config

        if self.climate_experiment is not None:
            with open(self.climate_timeseries_names_path, "r") as f:
                climate_timeseries_names = json.load(f)
            self.climate_timeseries_names = climate_timeseries_names

            # Load the climate array and apply weather transform
            climate_array = np.load(self.climate_array_path)
            weather_transform = WeatherStdNormalTransform(
                climate_array, channel_names=self.climate_timeseries_names
            )
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
            (
                self.seen_epw_training_set,
                self.seen_epw_validation_set,
                self.seen_epw_testing_set,
            ) = random_split(
                seen_epw_buiding_dataset,
                [0.9, 0.05, 0.05],
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
            self.unseen_epw_validation_set, self.unseen_epw_testing_set = random_split(
                unseen_epw_validation_set,
                [0.5, 0.5],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            self.train_dataset = BuildingDataset(
                space_config,
                climate_array=None,
                path=self.train_data_path,
                key="batch_results",
            )
            target_transform = self.train_dataset.fit_target_transform()
            unseen_dataset = BuildingDataset(
                space_config,
                climate_array=None,
                path=self.train_data_path,
                key="batch_results",
            )
            unseen_dataset.load_target_transform(target_transform)
            self.target_transform = target_transform
            self.val_dataset, self.test_dataset = random_split(
                unseen_dataset,
                [0.5, 0.5],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.seen_epw_training_set
            if self.climate_experiment is not None
            else self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return (
            [
                DataLoader(
                    self.seen_epw_validation_set,
                    batch_size=self.batch_size * self.val_batch_mult,
                ),
                DataLoader(
                    self.unseen_epw_validation_set,
                    batch_size=self.batch_size * self.val_batch_mult,
                ),
            ]
            if self.climate_experiment is not None
            else DataLoader(
                self.val_dataset,
                batch_size=self.batch_size * self.val_batch_mult,
            )
        )

    def test_dataloader(self):
        return (
            [
                DataLoader(
                    self.seen_epw_testing_set,
                    batch_size=self.batch_size * self.val_batch_mult,
                ),
                DataLoader(
                    self.unseen_epw_testing_set,
                    batch_size=self.batch_size * self.val_batch_mult,
                ),
            ]
            if self.climate_experiment is not None
            else DataLoader(
                self.test_dataset,
                batch_size=self.batch_size * self.val_batch_mult,
            )
        )


class BuildingHourlyDataModule(pl.LightningDataModule):
    """
    A LightningDataModule for the BuildingDataset, which abstracts loading
    and preparing the data for training, validation, and testing
    """

    def __init__(
        self,
        bucket: str = "ml-for-bem",
        remote_experiment: str = "full_climate_zone/v3",
        data_dir: str = "path/to/dir",
        climate_experiment: str = "weather/v1",
        batch_size: int = 32,
        val_batch_mult: int = 16,
    ):
        """
        Create a BuildingDataModule which can be used to load and prepare the data for training, validation, and testing.
        The data is stored on s3 and downloaded to the local data_dir if not there already

        Args:
            bucket: the name of the s3 bucket where the data is stored
            remote_experiment: the name of the experiment on s3
            data_dir: the local directory where the data should be stored
            climate_experiment: the name of the experiment on s3 where the climate data is stored
            batch_size (int): the batch size to use for training
            val_batch_mult (int): the multiplier to use on the training batch size to determine validation/testing batch size

        Returns:
            BuildingDataModule: a LightningDataModule which can be used to load and prepare the data for training, validation, and testing

        """
        super().__init__()

        # Store the arguments
        self.data_dir = data_dir
        self.bucket = bucket
        self.batch_size = batch_size
        self.val_batch_mult = val_batch_mult
        self.remote_experiment = remote_experiment
        self.climate_experiment = climate_experiment

        # Make the paths
        self.experiment_root = Path(data_dir) / remote_experiment
        self.train_data_path = Path(self.experiment_root) / "train" / "hourly.hdf"
        self.test_data_dir = Path(self.experiment_root) / "test" / "hourly.hdf"
        self.space_config_path = (
            Path(self.experiment_root) / "train" / "space_definition.json"
        )
        self.climate_experiment_root = (
            (Path(data_dir) / climate_experiment)
            if self.climate_experiment is not None
            else None
        )
        self.climate_array_path = (
            (self.climate_experiment_root / "global_climate_array.npy")
            if self.climate_experiment is not None
            else None
        )
        self.climate_timeseries_names_path = (
            (self.climate_experiment_root / "timeseries.json")
            if self.climate_experiment is not None
            else None
        )

    def prepare_data(self):
        # Download the data from s3 if it doesn't exist locally
        s3 = boto3.client("s3")
        if not os.path.exists(self.train_data_path):
            logger.info("downloading hourly data!")
            os.makedirs(self.train_data_path.parent, exist_ok=True)
            s3.download_file(
                self.bucket,
                f"{self.remote_experiment}/train/hourly.hdf",
                self.train_data_path,
            )
        if not os.path.exists(self.test_data_dir):
            os.makedirs(self.test_data_dir.parent, exist_ok=True)
            s3.download_file(
                self.bucket,
                f"{self.remote_experiment}/test/hourly.hdf",
                self.test_data_dir,
            )
        if not os.path.exists(self.space_config_path):
            os.makedirs(self.space_config_path.parent, exist_ok=True)
            s3.download_file(
                self.bucket,
                f"{self.remote_experiment}/train/space_definition.json",
                self.space_config_path,
            )
        if self.climate_experiment is not None:
            if not os.path.exists(self.climate_array_path):
                os.makedirs(self.climate_array_path.parent, exist_ok=True)
                s3.download_file(
                    self.bucket,
                    f"{self.climate_experiment}/global_climate_array.npy",
                    self.climate_array_path,
                )

            if not os.path.exists(self.climate_timeseries_names_path):
                os.makedirs(self.climate_timeseries_names_path.parent, exist_ok=True)
                s3.download_file(
                    self.bucket,
                    f"{self.climate_experiment}/timeseries.json",
                    self.climate_timeseries_names_path,
                )

    def setup(self, stage: str):
        # Load the data, apply transforms, make datasets

        # Load the space config definition
        with open(self.space_config_path, "r") as f:
            space_config = json.load(f)
        self.space_config = space_config

        if self.climate_experiment is not None:
            with open(self.climate_timeseries_names_path, "r") as f:
                climate_timeseries_names = json.load(f)
            self.climate_timeseries_names = climate_timeseries_names

            # Load the climate array and apply weather transform
            climate_array = np.load(self.climate_array_path)
            weather_transform = WeatherStdNormalTransform(
                climate_array, channel_names=self.climate_timeseries_names
            )
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
            seen_epw_buiding_dataset = BuildingHourlyDataset(
                space_config,
                self.climate_array,
                self.train_data_path,
                key="batch_results",
            )
            logger.info("Fitting target transform...")
            target_transform = seen_epw_buiding_dataset.fit_target_transform()
            self.target_transform = target_transform

            # Split the dataset for seen weather data into training and validation
            (
                self.seen_epw_training_set,
                self.seen_epw_validation_set,
                self.seen_epw_testing_set,
            ) = random_split(
                seen_epw_buiding_dataset,
                [0.9, 0.05, 0.05],
                generator=torch.Generator().manual_seed(42),
            )

            # Make the dataset for unseen weather data and load the target transform
            unseen_epw_validation_set = BuildingHourlyDataset(
                space_config,
                self.climate_array,
                self.test_data_dir,
                key="batch_results",
            )
            unseen_epw_validation_set.load_target_transform(target_transform)
            self.unseen_epw_validation_set, self.unseen_epw_testing_set = random_split(
                unseen_epw_validation_set,
                [0.5, 0.5],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            self.train_dataset = BuildingHourlyDataset(
                space_config,
                climate_array=None,
                path=self.train_data_path,
                key="batch_results",
            )
            target_transform = self.train_dataset.fit_target_transform()
            unseen_dataset = BuildingHourlyDataset(
                space_config,
                climate_array=None,
                path=self.train_data_path,
                key="batch_results",
            )
            unseen_dataset.load_target_transform(target_transform)
            self.target_transform = target_transform
            self.val_dataset, self.test_dataset = random_split(
                unseen_dataset,
                [0.5, 0.5],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.seen_epw_training_set
            if self.climate_experiment is not None
            else self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return (
            [
                DataLoader(
                    self.seen_epw_validation_set,
                    batch_size=self.batch_size * self.val_batch_mult,
                ),
                DataLoader(
                    self.unseen_epw_validation_set,
                    batch_size=self.batch_size * self.val_batch_mult,
                ),
            ]
            if self.climate_experiment is not None
            else DataLoader(
                self.val_dataset,
                batch_size=self.batch_size * self.val_batch_mult,
            )
        )

    def test_dataloader(self):
        return (
            [
                DataLoader(
                    self.seen_epw_testing_set,
                    batch_size=self.batch_size * self.val_batch_mult,
                ),
                DataLoader(
                    self.unseen_epw_testing_set,
                    batch_size=self.batch_size * self.val_batch_mult,
                ),
            ]
            if self.climate_experiment is not None
            else DataLoader(
                self.test_dataset,
                batch_size=self.batch_size * self.val_batch_mult,
            )
        )


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
