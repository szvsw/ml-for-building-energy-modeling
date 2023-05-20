from typing import List

import logging
import os
from pathlib import Path
import numpy as np
import h5py

# import torch

from schedules import mutate_timeseries
from storage import download_from_bucket
from schema import Schema


logging.basicConfig()
logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)

root_dir = Path(os.path.abspath(os.path.dirname(__file__)))

DATA_PATH = root_dir / "data"

HOURS_IN_YEAR = 8760

"""Helpers"""


def normalize(data, maxv, minv):
    return (data - minv) / (maxv - minv)


class ClimateData:
    # Weather data
    config = {
        "dbt": {
            "max": 50,  # 50.5
            "min": -35,
        },
        "rh": {
            "max": 100,
            "min": 0.0,  # 2.0
        },
        "atm": {
            "max": 105800.0,
            "min": 75600.0,
        },
        "ghrad": {
            "max": 1200,  # 1154
            "min": 0.0,
        },
        "dnrad": {
            "max": 1097,
            "min": 0.0,
        },
        "dhrad": {
            "max": 689,
            "min": 0,
        },
        "skyt": {
            "max": 32.3,
            "min": -58.3,
        },
        "tsol": {
            "max": 60,
            "min": -40,
        },
    }

    series_ix = {key: ix for ix, key in enumerate(config.keys())}
    series_maxes = [bounds["max"] for bounds in config.values()]
    series_mins = [bounds["min"] for bounds in config.values()]

    tsol_path = DATA_PATH / "epws" / "tsol.npy"
    climate_arr_path = DATA_PATH / "epws" / "climate_array.npy"

    __slots__ = ("tsol_arr", "climate_arr", "tsol_arr_norm", "climate_arr_norm")

    def __init__(self) -> None:
        try:
            logging.info("Loading TSol Array file...")
            self.tsol_arr = np.load(ClimateData.tsol_path)
        except FileNotFoundError:
            logging.info("Could not find TSol Array file.  Will attempt to download...")
            download_from_bucket("tsol.npy", ClimateData.tsol_path)
            self.tsol_arr = np.load(ClimateData.tsol_path)
        logging.info(f"Loaded TSol Array. Shape={self.tsol_arr.shape}")

        """AXES: 0 - city, 1 - channel, 2 - time"""
        try:
            logging.info("Loading Climate Array file...")
            self.climate_arr = np.load(ClimateData.climate_arr_path)
        except FileNotFoundError:
            logging.info(
                "Could not find Climate Array file. Will attempt to download..."
            )
            download_from_bucket("climate_array.npy", ClimateData.climate_arr_path)
            self.climate_arr = np.load(ClimateData.climate_arr_path)
        logging.info(f"Loaded EPW Climate Array.  Shape={self.climate_arr.shape}")

        logging.info("Normalizing climate data...")
        # TODO: Vectorize this normalization
        self.climate_arr_norm = np.zeros(self.climate_arr.shape)
        self.tsol_arr_norm = np.zeros(self.tsol_arr.shape)
        for j in range(self.climate_arr.shape[0]):
            for i in range(7):
                # Handle normal time series
                self.climate_arr_norm[j, i, :] = normalize(
                    self.climate_arr[j, i, :],
                    ClimateData.series_maxes[i],
                    ClimateData.series_mins[i],
                )
            for i in range(4):
                # Handle stacked orientation time series
                self.tsol_arr_norm[j, i, :] = normalize(
                    self.tsol_arr[j, i, :],
                    ClimateData.config["tsol"]["max"],
                    ClimateData.config["tsol"]["min"],
                )
        logging.info("All climate data successfully loaded!")


class DataManager:
    __slots__ = (
        "schema",
        "climate_data",
        "results",
        "full_storage_batch",
        "default_schedules",
    )

    folder = DATA_PATH / "model_data_manager"
    os.makedirs(folder, exist_ok=True)
    default_schedules_path = folder / "default_schedules.npy"
    full_storage_batch_path = folder / "all_input_batches.hdf5"
    full_results_path = folder / "all_data_monthly.hdf5"

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        if not os.path.exists(DataManager.full_results_path):
            os.makedirs(DATA_PATH, exist_ok=True)
            logger.info(
                "Full monthly data set not found.  Downloading from GCloud Bucket..."
            )
            download_from_bucket("all_data_monthly.hdf5", DataManager.full_results_path)
            logger.info("Done downloading dataset!")

        try:
            logging.info("Loading Default Schedules file...")
            self.default_schedules = np.load(DataManager.default_schedules_path)
        except FileNotFoundError:
            logging.info(
                "Could not find Default Schedules file. Will attempt to download..."
            )
            download_from_bucket(
                "default_schedules.npy", DataManager.default_schedules_path
            )
            self.default_schedules = np.load(DataManager.default_schedules_path)
        # Schedules were stored as Res_Occ, Res_Light Res_Equip, but schema expects order to be Equip, Lights, Occ
        self.default_schedules = np.flip(self.default_schedules, axis=0)
        logging.info(
            f"Loaded Default Schedules Array.  Shape={self.default_schedules.shape}"
        )

        logger.info("Loading the full dataset into main RAM...")
        self.results = {}
        with h5py.File(DataManager.full_results_path, "r") as f:
            self.results["area"] = f["area"][...].reshape(-1, 1)
            self.results["total_heating"] = f["total_heating"][...].reshape(
                -1, 1
            )  # this loads the whole batch into memory!
            self.results["total_cooling"] = f["total_cooling"][...].reshape(
                -1, 1
            )  # this loads the whole batch into memory!
            self.results["window_u"] = f["window_u"][...].reshape(-1, 1)
            self.results["facade_hcp"] = f["facade_hcp"][...].reshape(-1, 1)
            self.results["roof_hcp"] = f["roof_hcp"][...].reshape(-1, 1)
            self.results["monthly"] = f["monthly"][
                ...
            ]  # this loads the whole batch into memory!
            self.full_storage_batch = f["storage_batch"][...]

        logger.info("Finished loading the full dataset.")
        logger.info(
            f"Full Input Batch Size (in storage form, not MLVec Form): {self.full_storage_batch.nbytes / 1000000}MB"
        )
        logger.info("Loading climate data...")
        self.climate_data = ClimateData()
        logger.info("Finished loading climate data.")

    def get_batch_climate_timeseries(self, batch):
        orientations = (
            self.schema["orientation"].extract_storage_values_batch(batch).astype(int)
        )
        epw_idxs = (
            self.schema["base_epw"].extract_storage_values_batch(batch).astype(int)
        )
        epw_data = self.climate_data.climate_arr_norm[epw_idxs].squeeze()
        tsol_data = self.climate_data.tsol_arr_norm[
            epw_idxs.flatten(), orientations.flatten()
        ].reshape(-1, 1, HOURS_IN_YEAR)
        climate_timeseries_vector = np.concatenate((epw_data, tsol_data), axis=1)
        return climate_timeseries_vector

    # TODO: use new uvalue/facade etc
    def get_building_vector(self, batch):
        bldg_params, timeseries_ops = self.schema.to_ml(batch)
        return bldg_params

    def make_schedules(self, batch):
        timeseries_ops = self.schema["schedules"].extract_storage_values_batch(batch)
        seeds = self.schema["schedules_seed"].extract_storage_values_batch(batch)
        schedules = []
        # Sad for loop
        for i, seed in enumerate(seeds):
            seed = int(seed)
            ops = timeseries_ops[i]
            scheds = mutate_timeseries(self.default_schedules, ops, seed)
            schedules.append(scheds)
        schedules = np.stack(schedules)
        return schedules

    def make_permutation_filepath(self, name_prefix, id):
        return DataManager.folder / f"{name_prefix}-PERM{id:02d}.hdf5"

    def make_batch_permutations(
        self, input_batch, targets, permutation_count, name_prefix
    ):
        logger.info("Making schedules... this may take a while...")
        schedules = self.make_schedules(input_batch)
        bldg_param_ml_vectors, _ = self.schema.to_ml(input_batch)
        og_batch_id = self.schema["batch_id"].extract_storage_values_batch(input_batch)
        og_var_id = self.schema["variation_id"].extract_storage_values_batch(
            input_batch
        )
        epw_id = self.schema["base_epw"].extract_storage_values_batch(input_batch)
        orientations = self.schema["orientation"].extract_storage_values_batch(
            input_batch
        )
        logger.info("Making dataset permutations...")
        for permutation_id in range(permutation_count):
            logger.info(f"Making permutation {permutation_id}...")
            perm_order = np.arange(input_batch.shape[0])
            np.random.shuffle(perm_order)
            shuffled_ml_vecs = bldg_param_ml_vectors[perm_order]
            shuffled_scheds = schedules[perm_order]
            shuffled_targets = targets[perm_order]
            shuffled_batch_ids = og_batch_id[
                perm_order
            ]  # how to find the original data if needed
            shuffled_var_ids = og_var_id[perm_order]
            shuffled_epw_ids = epw_id[perm_order]
            shuffled_orientations = orientations[perm_order]
            logger.info(f"Saving permutation {permutation_id}...")
            filepath = self.make_permutation_filepath(name_prefix, permutation_id)
            with h5py.File(filepath, "w") as f:
                for (name, data) in [
                    ("ml_vectors", shuffled_ml_vecs),
                    ("schedules", shuffled_scheds),
                    ("targets", shuffled_targets),
                    ("batch_ids", shuffled_batch_ids),
                    ("var_ids", shuffled_var_ids),
                    ("epw_ids", shuffled_epw_ids),
                    ("orientations", shuffled_orientations),
                ]:
                    f.create_dataset(
                        name=name,
                        shape=data.shape,
                        dtype=data.dtype,
                        data=data,
                        compression="lzf",
                        chunks=True,
                    )

    def load_minibatch(self, name_prefix, file_id, start, n):
        filepath = self.make_permutation_filepath(name_prefix, file_id)
        data = {
            "parent": self,
            "start": start,
            "n": n,
            "name_prefix": name_prefix,
            "file_id": file_id,
        }
        with h5py.File(filepath, "r") as f:
            for key in [
                "ml_vectors",
                "schedules",
                "targets",
                "epw_ids",
                "orientations",
            ]:
                data[key] = f[key][start : start + n]
        return MLMiniBatch(**data)


class MLMiniBatch:

    __slots__ = (
        "start",
        "n",
        "name_prefix",
        "file_id",
        "ml_vectors",
        "schedules",
        "targets",
        "batch_ids",
        "var_ids",
        "epw_ids",
        "orientations",
        "climate_vectors",
        "parent",
    )

    def __init__(
        self,
        ml_vectors,
        schedules,
        targets,
        epw_ids,
        orientations,
        parent,
        start,
        n,
        name_prefix,
        file_id,
    ) -> None:
        self.ml_vectors = ml_vectors
        self.schedules = schedules
        self.targets = targets
        self.epw_ids = epw_ids.astype(int).flatten()
        self.orientations = orientations.astype(int).flatten()
        self.parent: DataManager = parent
        self.start = start
        self.n = n
        self.name_prefix = name_prefix
        self.file_id = file_id
        self.load_climate_vectors()

    @property
    def filepath(self):
        return self.parent.make_permutation_filepath(self.name_prefix, self.file_id)

    @property
    def size(self):
        return self.ml_vectors.shape[0]

    def load_climate_vectors(self):
        epw_data = self.parent.climate_data.climate_arr_norm[self.epw_ids].squeeze()
        tsol_data = self.parent.climate_data.tsol_arr_norm[
            self.epw_ids, self.orientations
        ].reshape(-1, 1, HOURS_IN_YEAR)
        self.climate_vectors = np.concatenate((epw_data, tsol_data), axis=1)

    @property
    def timeseries(self):
        return np.concatenate((self.climate_vectors, self.schedules), axis=1)
