import logging
import os
from typing import List, Literal
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from networks import EnergyCNN, MonthlyEnergyCNN
from schedules import mutate_timeseries
from storage import download_from_bucket, upload_to_bucket
from schema import Schema, OneHotParameter, WindowParameter

from tqdm.autonotebook import tqdm


logging.basicConfig()
logger = logging.getLogger("Surrogate")
logger.setLevel(logging.INFO)

root_dir = Path(os.path.abspath(os.path.dirname(__file__)))
checkpoints_dir = root_dir / "checkpoints"

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = root_dir / "data"

HOURS_IN_YEAR = 8760


def normalize(data, maxv, minv):
    return (data - minv) / (maxv - minv)


class ClimateData:
    # Weather data
    config = {
        "dbt": {
            "max": 50,
            "min": -35,
            "description": "Dry Bulb Temperature",
        },  # 50.5
        "rh": {
            "max": 100,
            "min": 0.0,
            "description": "Relative Humidity",
        },  # 2.0
        "atm": {
            "max": 105800.0,
            "min": 75600.0,
            "description": "Atmospheric Pressure",
        },
        "ghrad": {
            "max": 1200,  # 1154
            "min": 0.0,
            "description": "Global Horizontal Radiation",
        },
        "dnrad": {
            "max": 1097,
            "min": 0.0,
            "description": "Direct Normal Radiation",
        },
        "dhrad": {
            "max": 689,
            "min": 0,
            "description": "Direct Horizontal Radiation",
        },
        "skyt": {
            "max": 32.3,
            "min": -58.3,
            "description": "Sky Temperature",
        },
        "tsol": {
            "max": 60,
            "min": -40,
            "description": "T-SolAir",
        },
    }

    series_ix = {key: ix for ix, key in enumerate(config.keys())}
    series_maxes = [bounds["max"] for bounds in config.values()]
    series_mins = [bounds["min"] for bounds in config.values()]

    tsol_path = DATA_PATH / "epws" / "tsol.npy"
    climate_arr_path = DATA_PATH / "epws" / "climate_array.npy"

    __slots__ = (
        "tsol_arr",
        "climate_arr",
        "tsol_arr_norm",
        "climate_arr_norm",
    )

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


# TODO: consider updating roof/facade hcp values with true/adjusted values
# TODO: deal with windows (currently using sample values rather than archetypal vals)
# TODO: use true zone area for normalization?
# TODO: check perim/core orders
class Surrogate:
    __slots__ = (
        "schema",
        "climate_data",
        "results",
        "full_storage_batch",
        "default_schedules",
        "eui_perim_heating_max",
        "eui_perim_heating_min",
        "eui_perim_cooling_max",
        "eui_perim_cooling_min",
        "eui_core_heating_max",
        "eui_core_heating_min",
        "eui_core_cooling_max",
        "eui_core_cooling_min",
        "area_max",
        "area_min",
        "area_core_max",
        "area_core_min",
        "area_perim_max",
        "area_perim_min",
        "building_params_per_vector",
        "timeseries_per_vector",
        "timeseries_per_output",
        "output_resolution",
        "latent_size",
        "energy_cnn_in_size",
        "timeseries_net",
        "energy_net",
        "loss_fn",
        "optimizer",
        "learning_rate",
        "training_loss_history",
        "validation_loss_history",
        "withheld_loss_history",
        "latentvect_history",
    )

    folder = DATA_PATH / "model_data_manager"
    os.makedirs(folder, exist_ok=True)
    default_schedules_path = folder / "default_schedules.npy"
    full_storage_batch_path = folder / "all_input_batches.hdf5"
    full_results_path = folder / "all_data_monthly.hdf5"

    def __init__(
        self,
        schema: Schema,
        learning_rate=1e-3,
        checkpoint=None,
        load_training_data=True,
        cpu=False,
    ) -> None:
        global device
        if cpu:
            device = "cpu"
        logger.info(f"Using {device} for surrogate model.")
        self.schema = schema
        if load_training_data:
            self.load_all_training_data()
        self.config_network_dims(
            dim_source="checkpoint" if checkpoint is not None else "training",
            checkpoint=checkpoint,
        )
        logger.info(
            f"{self.building_params_per_vector} building parameters per input vector"
        )
        logger.info(f"{self.timeseries_per_vector} timeseries per input vector")
        logger.info(f"{self.timeseries_per_output} timeseries per output vector")
        logger.info(f"{self.output_resolution} timesteps in output.")

        logger.info("Initializing machine learning objects...")
        self.timeseries_net = MonthlyEnergyCNN(
            in_channels=self.timeseries_per_vector, out_channels=self.latent_size
        ).to(device)
        self.energy_net = EnergyCNN(
            in_channels=self.energy_cnn_in_size, out_channels=self.timeseries_per_output
        ).to(device)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            list(self.timeseries_net.parameters()) + list(self.energy_net.parameters()),
            lr=self.learning_rate,
        )
        self.training_loss_history = []
        self.validation_loss_history = []
        self.withheld_loss_history = []
        self.latentvect_history = []
        logger.info("ML objects initialized.")

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def load_all_training_data(self):
        if not os.path.exists(Surrogate.full_results_path):
            os.makedirs(DATA_PATH, exist_ok=True)
            logger.info(
                "Full monthly data set not found.  Downloading from GCloud Bucket..."
            )
            download_from_bucket("all_data_monthly.hdf5", Surrogate.full_results_path)
            logger.info("Done downloading dataset!")

        try:
            logging.info("Loading Default Schedules file...")
            self.default_schedules = np.load(Surrogate.default_schedules_path)
        except FileNotFoundError:
            logging.info(
                "Could not find Default Schedules file. Will attempt to download..."
            )
            download_from_bucket(
                "default_schedules.npy", Surrogate.default_schedules_path
            )
            self.default_schedules = np.load(Surrogate.default_schedules_path)
        # Schedules were stored as Res_Occ, Res_Light Res_Equip, but schema expects order to be Equip, Lights, Occ
        self.default_schedules = np.flip(self.default_schedules, axis=0)
        logging.info(
            f"Loaded Default Schedules Array.  Shape={self.default_schedules.shape}"
        )

        logger.info("Loading the full dataset into main RAM...")
        self.results = {}
        with h5py.File(Surrogate.full_results_path, "r") as f:
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
            self.results["area_core"] = self.results["area"] * (
                1
                - self.schema["perim_2_footprint"].extract_storage_values_batch(
                    self.full_storage_batch
                )
            )
            self.results["area_perim"] = self.results["area"] * self.schema[
                "perim_2_footprint"
            ].extract_storage_values_batch(self.full_storage_batch)
            perim_heating_eui = (
                self.results["monthly"][:, 0] * 2.7777e-7 / self.results["area_perim"]
            )
            perim_cooling_eui = (
                self.results["monthly"][:, 1] * 2.7777e-7 / self.results["area_perim"]
            )
            core_heating_eui = (
                self.results["monthly"][:, 2] * 2.7777e-7 / self.results["area_core"]
            )
            core_cooling_eui = (
                self.results["monthly"][:, 3] * 2.7777e-7 / self.results["area_core"]
            )
            eui_unnorm = np.stack(
                [
                    perim_heating_eui,
                    perim_cooling_eui,
                    core_heating_eui,
                    core_cooling_eui,
                ],
                axis=1,
            )
            perim_heating_eui_max = np.max(perim_heating_eui)
            perim_heating_eui_min = np.min(perim_heating_eui)
            perim_cooling_eui_max = np.max(perim_cooling_eui)
            perim_cooling_eui_min = np.min(perim_cooling_eui)
            core_heating_eui_max = np.max(core_heating_eui)
            core_heating_eui_min = np.min(core_heating_eui)
            core_cooling_eui_max = np.max(core_cooling_eui)
            core_cooling_eui_min = np.min(core_cooling_eui)
            self.eui_perim_heating_max = perim_heating_eui_max
            self.eui_perim_heating_min = perim_heating_eui_min
            self.eui_perim_cooling_max = perim_cooling_eui_max
            self.eui_perim_cooling_min = perim_cooling_eui_min
            self.eui_core_heating_max = core_heating_eui_max
            self.eui_core_heating_min = core_heating_eui_min
            self.eui_core_cooling_max = core_cooling_eui_max
            self.eui_core_cooling_min = core_cooling_eui_min
            perim_heating_eui_norm = normalize(
                perim_heating_eui, perim_heating_eui_max, perim_heating_eui_min
            )
            perim_cooling_eui_norm = normalize(
                perim_cooling_eui, perim_cooling_eui_max, perim_cooling_eui_min
            )
            core_heating_eui_norm = normalize(
                core_heating_eui, core_heating_eui_max, core_heating_eui_min
            )
            core_cooling_eui_norm = normalize(
                core_cooling_eui, core_cooling_eui_max, core_cooling_eui_min
            )
            eui_norm = np.stack(
                [
                    perim_heating_eui_norm,
                    perim_cooling_eui_norm,
                    core_heating_eui_norm,
                    core_cooling_eui_norm,
                ],
                axis=1,
            )
            self.results["eui_unnormalized"] = eui_unnorm
            self.results["eui_normalized"] = eui_norm
        self.area_core_max = np.max(self.results["area_core"])
        self.area_core_min = np.min(self.results["area_core"])
        self.area_perim_max = np.max(self.results["area_perim"])
        self.area_perim_min = np.min(self.results["area_perim"])
        self.area_max = np.max(self.results["area"])
        self.area_min = np.min(self.results["area"])

        logger.info("Finished loading the full dataset.")
        logger.info(
            f"Full Input Batch Size (in storage form, not MLVec Form): {self.full_storage_batch.nbytes / 1000000}MB"
        )
        logger.info("Loading climate data...")
        self.climate_data = ClimateData()
        logger.info("Finished loading climate data.")

    def config_network_dims(
        self,
        dim_source: Literal["training", "checkpoint"] = "training",
        checkpoint=None,
    ):
        if dim_source == "training":
            self.configure_network_dims_from_training_data()
        elif dim_source == "checkpoint":
            checkpoint_path = checkpoints_dir / checkpoint
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoints_dir / checkpoint.split("/")[0], exist_ok=True)
                download_from_bucket("models/" + checkpoint, checkpoint_path)
            data = torch.load(checkpoint_path)
            self.building_params_per_vector = data["building_params_per_vector"]
            self.timeseries_per_vector = data["timeseries_per_vector"]
            self.timeseries_per_output = data["timeseries_per_output"]
            self.output_resolution = data["output_resolution"]
            self.latent_size = data["latent_size"]
            self.energy_cnn_in_size = data["energy_cnn_in_size"]

    def configure_network_dims_from_training_data(self):
        logger.info("Checking model dimensions...")
        level = logger.level
        logger.setLevel(logging.ERROR)
        bv, tv, results = self.make_dataset(start_ix=0, count=10)
        self.building_params_per_vector = bv.shape[1]
        self.timeseries_per_vector = tv.shape[1]
        self.timeseries_per_output = results.shape[1]
        self.output_resolution = results.shape[-1]
        self.latent_size = self.building_params_per_vector
        self.energy_cnn_in_size = self.latent_size + self.building_params_per_vector
        logger.setLevel(level)

    def load_checkpoint(self, checkpoint):
        # TODO: implement full surrogate config from checkpoint
        # TODO: implement detection of latest checkpoint available
        checkpoint_path = checkpoints_dir / checkpoint
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoints_dir / checkpoint.split("/")[0], exist_ok=True)
            download_from_bucket("models/" + checkpoint, checkpoint_path)
        data = torch.load(checkpoint_path)
        timeseries_net_dict = data["timeseries_net_state_dict"]
        energy_net_dict = data["energy_net_state_dict"]
        self.energy_net.load_state_dict(energy_net_dict)
        self.timeseries_net.load_state_dict(timeseries_net_dict)
        self.training_loss_history = data["training_loss_history"]
        self.withheld_loss_history = data["withheld_loss_history"]
        self.validation_loss_history = data["validation_loss_history"]

    def get_batch_climate_timeseries(self, batch):
        # TODO: figure out why this takes so long
        logger.info("Constructing climate timeseries for batch...")
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
        logger.info("Climate timeseries for batch constructed.")
        return climate_timeseries_vector

    # TODO: use new uvalue/facade etc
    def get_batch_building_vector(self, batch):
        logger.info("Constructing building vector for batch...")
        bldg_params, timeseries_ops = self.schema.to_ml(batch)
        logger.info("Building vector for batch constructed.")
        return bldg_params

    def get_batch_schedules(self, batch):
        logger.info("Constructing schedules for batch...")
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
        logger.info("Schedules for batch constructed.")
        return schedules

    def make_dataset(self, start_ix, count):
        logger.info("Constructing dataset...")
        areas = self.results["area"][start_ix : start_ix + count]
        areas_normalized = normalize(areas, self.area_max, self.area_min)
        perim_areas = self.results["area_perim"][start_ix : start_ix + count]
        core_areas = self.results["area_core"][start_ix : start_ix + count]
        perim_areas_norm = normalize(
            perim_areas, self.area_perim_max, self.area_perim_min
        )
        core_areas_norm = normalize(core_areas, self.area_core_max, self.area_core_min)

        batch = self.full_storage_batch[start_ix : start_ix + count]
        bldg_params = self.get_batch_building_vector(batch)
        building_vector = np.concatenate(
            [bldg_params, areas_normalized, perim_areas_norm, core_areas_norm], axis=1
        )

        climate_timeseries = self.get_batch_climate_timeseries(batch)
        schedules = self.get_batch_schedules(batch)
        timeseries_vector = np.concatenate([climate_timeseries, schedules], axis=1)

        # loads = self.results["eui"][start_ix:start_ix+count]
        # loads_normalized = normalize(loads, self.eui_max, self.eui_min)
        loads_normalized = self.results["eui_normalized"][start_ix : start_ix + count]

        logger.info("Dataset constructed.")
        return building_vector, timeseries_vector, loads_normalized

    def make_dataloader(self, start_ix, count, dataloader_batch_size):
        building_vector, timeseries_vector, loads_normalized = self.make_dataset(
            start_ix, count
        )
        torch.cuda.empty_cache()

        logger.info("Building dataloaders...")
        dataset = {}
        for i in range(building_vector.shape[0]):
            # DICT ENTRIES MUST BE IN ORDER
            dataset[i] = dict(
                {
                    "building_vector": np.array(
                        [building_vector[i]] * self.output_resolution
                    ).T,
                    "timeseries_vector": timeseries_vector[i],
                    "results_vector": loads_normalized[i],
                }
            )
        generator = torch.Generator()
        generator.manual_seed(0)

        train, val, test = torch.utils.data.random_split(
            dataset, lengths=[0.8, 0.1, 0.1], generator=generator
        )
        training_dataloader = torch.utils.data.DataLoader(
            train, batch_size=dataloader_batch_size, shuffle=False
        )
        validation_dataloader = torch.utils.data.DataLoader(
            val, batch_size=dataloader_batch_size, shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            test, batch_size=dataloader_batch_size, shuffle=False
        )
        logger.info("Dataloaders built.")
        return {
            "datasets": {"train": train, "test": test, "validate": val},
            "dataloaders": {
                "train": training_dataloader,
                "test": test_dataloader,
                "validate": validation_dataloader,
            },
        }

    def train(
        self,
        run_name,
        train_test_split_ix=400000,
        n_full_epochs=3,
        n_mini_epochs=3,
        mini_epoch_batch_size=50000,
        dataloader_batch_size=200,
        step_loss_frequency=50,
        lr_schedule=None,
    ):
        # TODO: implement annual regularizer / possibly adaptive loss fns
        assert (
            train_test_split_ix % mini_epoch_batch_size == 0
        ), "The train/test split ix must be divisible by the mini epoch batch size."
        assert (
            mini_epoch_batch_size % dataloader_batch_size == 0
        ), "The dataloader batch size must be a factor of the minibatch size"
        self.energy_net.train()
        self.timeseries_net.train()
        final_start_ix = train_test_split_ix - mini_epoch_batch_size
        unseen_testing_cities = self.make_dataloader(
            train_test_split_ix + 50000,
            count=20000,
            dataloader_batch_size=dataloader_batch_size,
        )

        for full_epoch_num in range(n_full_epochs):
            if lr_schedule is not None:
                for group in self.optimizer.param_groups:
                    group["lr"] = lr_schedule[full_epoch_num]
            logger.info(f"\n\n\n {'-'*20} MAJOR Epoch {full_epoch_num} {'-'*20}")
            for start_idx in range(0, final_start_ix + 1, mini_epoch_batch_size):
                logger.info(
                    f"{'-'*15} BATCH {start_idx:05d}:{(start_idx+mini_epoch_batch_size):05d}{'-'*15}"
                )
                data = self.make_dataloader(
                    start_ix=start_idx,
                    count=mini_epoch_batch_size,
                    dataloader_batch_size=dataloader_batch_size,
                )
                training_dataloader = data["dataloaders"]["train"]
                validation_dataloader = data["dataloaders"]["validate"]

                logger.info("Starting batch training...")

                for epoch_num in range(n_mini_epochs):
                    logger.info(f"{'-'*20} MiniBatch Epoch number {epoch_num} {'-'*20}")
                    self.timeseries_net.train()
                    self.energy_net.train()
                    for j, sample in enumerate(training_dataloader):
                        self.optimizer.zero_grad()
                        projection_results = self.project_dataloader_sample(sample)
                        loss = projection_results["loss"]
                        latvect = projection_results["timeseries_latvect"]
                        if j % step_loss_frequency == 0:
                            logger.info(f"Step {j} loss: {loss.item()}")
                            self.latentvect_history.append(latvect.detach())
                        self.training_loss_history.append(
                            [len(self.training_loss_history), loss.item()]
                        )
                        loss.backward()
                        self.optimizer.step()

                    self.timeseries_net.eval()
                    self.energy_net.eval()
                    with torch.no_grad():
                        epoch_validation_loss = []
                        for sample in validation_dataloader:
                            projection_results = self.project_dataloader_sample(sample)
                            loss = projection_results["loss"]
                            epoch_validation_loss.append(loss.item())
                        mean_validation_loss = np.mean(epoch_validation_loss)
                        logger.info(
                            f"Mean validation loss for batch: {mean_validation_loss}"
                        )

                        self.validation_loss_history.append(
                            [len(self.training_loss_history), mean_validation_loss]
                        )

                # Finished repeating training on MiniBatch, check loss on fully unseen cities
                logger.info("Computing loss on withheld climate zone data...")
                epoch_validation_loss = []
                self.timeseries_net.eval()
                self.energy_net.eval()
                with torch.no_grad():
                    for sample in unseen_testing_cities["dataloaders"][
                        "train"
                    ]:  # using train is fine since this data is never seen
                        projection_results = self.project_dataloader_sample(sample)
                        loss = projection_results["loss"]
                        epoch_validation_loss.append(loss.item())
                    mean_validation_loss = np.mean(epoch_validation_loss)
                    logger.info(
                        f"Mean validation loss for withheld climate zone data: {mean_validation_loss}"
                    )
                    self.withheld_loss_history.append(
                        [len(self.training_loss_history), mean_validation_loss]
                    )

                self.plot_loss_histories()

                del training_dataloader
                del validation_dataloader
                del data

                self.save_checkpoint(
                    run_name,
                    epoch_num=full_epoch_num,
                    batch_start=start_idx,
                    train_test_split_idx=400000,
                    n_full_epochs=n_full_epochs,
                    n_mini_epochs=n_mini_epochs,
                    mini_epoch_batch_size=mini_epoch_batch_size,
                    dataloader_batch_size=dataloader_batch_size,
                )

    def project_dataloader_sample(self, sample, compute_loss=True):
        # Get the data
        timeseries_val = sample["timeseries_vector"].to(device).float()
        bldg_vect_val = sample["building_vector"].to(device).float()
        loads = sample["results_vector"].to(device).float() if compute_loss else None
        # Project timeseries to latent space
        timeseries_latvect_val = self.timeseries_net(timeseries_val)
        # Concatenate latent vector with building vector
        x_val = torch.cat([timeseries_latvect_val, bldg_vect_val], axis=1).squeeze(1)
        # Predict and compute loss
        predicted_loads = self.energy_net(x_val)
        # annual_pred = torch.sum(predicted_loads, axis=2)
        # annual_true = torch.sum(loads, axis=2)
        # TODO: implement adaptive weighting?
        loss = (
            self.loss_fn(predicted_loads, loads) if compute_loss else None
        )  # + 0.1*self.loss_fn(annual_pred, annual_true)
        return {
            "timeseries_latvect": timeseries_latvect_val,
            "predicted_loads": predicted_loads,
            "loads": loads,
            "loss": loss,
        }

    def save_checkpoint(
        self,
        run_name,
        epoch_num,
        batch_start,
        train_test_split_idx,
        n_full_epochs,
        n_mini_epochs,
        mini_epoch_batch_size,
        dataloader_batch_size,
    ):
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        timeseries_dict = self.timeseries_net.state_dict()
        energy_dict = self.energy_net.state_dict()
        optim_dict = self.optimizer.state_dict()
        checkpoint = {
            "timeseries_net_state_dict": timeseries_dict,
            "energy_net_state_dict": energy_dict,
            "optimizer_state_dict": optim_dict,
            "eui_perim_heating_max": self.eui_perim_heating_max,
            "eui_perim_heating_min": self.eui_perim_heating_min,
            "eui_perim_cooling_max": self.eui_perim_cooling_max,
            "eui_perim_cooling_min": self.eui_perim_cooling_min,
            "eui_core_heating_max": self.eui_core_heating_max,
            "eui_core_heating_min": self.eui_core_heating_min,
            "eui_core_cooling_max": self.eui_core_cooling_max,
            "eui_core_cooling_min": self.eui_core_cooling_min,
            "area_max": self.area_max,
            "area_min": self.area_min,
            "building_params_per_vector": self.building_params_per_vector,
            "timeseries_per_vector": self.timeseries_per_vector,
            "timeseries_per_output": self.timeseries_per_output,
            "output_resolution": self.output_resolution,
            "latent_size": self.latent_size,
            "energy_cnn_in_size": self.energy_cnn_in_size,
            "training_loss_history": self.training_loss_history,
            "validation_loss_history": self.validation_loss_history,
            "withheld_loss_history": self.withheld_loss_history,
            "learning_rate": self.learning_rate,
            "train_test_split_idx": train_test_split_idx,
            "n_full_epochs": n_full_epochs,
            "n_mini_epochs": n_mini_epochs,
            "mini_epoch_batch_size": mini_epoch_batch_size,
            "loader_batch_size": dataloader_batch_size,
            "epoch": epoch_num,
        }
        filename = f"{run_name}_{timestamp}_{epoch_num:03d}_{batch_start:05d}.pt"
        path = checkpoints_dir / filename
        torch.save(checkpoint, path)
        upload_to_bucket(f"models/{run_name}/{filename}", path)

    def evaluate_over_range(self, start_ix, count, segment="test"):
        true_loads = []
        pred_loads = []
        all_losses = []
        batch_size = 1000
        start_idxs = list(range(start_ix, start_ix + count, batch_size))
        for it in tqdm(range(len(start_idxs))):
            idx = start_idxs[it]
            losses = []
            epws = []
            czs = []
            temps = []
            data_to_plot = self.make_dataloader(
                start_ix=idx, count=1000, dataloader_batch_size=100
            )
            test_dataloader = data_to_plot["dataloaders"][segment]
            with torch.no_grad():
                for test_samples in test_dataloader:
                    projection_results = self.project_dataloader_sample(test_samples)
                    true_loads.append(projection_results["loads"])
                    pred_loads.append(projection_results["predicted_loads"])
                    all_losses.append(projection_results["loss"])

        true_loads = torch.vstack(true_loads)
        pred_loads = torch.vstack(pred_loads)
        return true_loads, pred_loads

    def plot_model_comparisons(
        self, start_ix, count, segment="test", plot_count=3, ylim=[-0.01, 0.5]
    ):
        self.energy_net.eval()
        self.timeseries_net.eval()
        level = logger.level
        logger.setLevel(logging.ERROR)
        true_loads, pred_loads = self.evaluate_over_range(
            start_ix=start_ix, count=count, segment=segment
        )
        ix = np.random.choice(np.arange(true_loads.shape[0]), plot_count, replace=False)
        for i in ix:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            plt.suptitle("Model Comparison")
            axs[0].plot(true_loads[i, 0].cpu(), label="Heating (true)")
            axs[0].plot(true_loads[i, 1].cpu(), label="Cooling (true)")
            axs[1].plot(true_loads[i, 2].cpu(), label="Heating (true)")
            axs[1].plot(true_loads[i, 3].cpu(), label="Cooling (true)")
            axs[0].plot(pred_loads[i, 0].cpu(), "-o", label="Heating (predicted)")
            axs[0].plot(pred_loads[i, 1].cpu(), "-o", label="Cooling (predicted)")
            axs[1].plot(pred_loads[i, 2].cpu(), "-o", label="Heating (predicted)")
            axs[1].plot(pred_loads[i, 3].cpu(), "-o", label="Cooling (predicted)")
            axs[0].set_ylim(ylim)
            axs[1].set_ylim(ylim)
            axs[0].set_title("Perimeter")
            axs[1].set_title("Core")
            axs[0].legend()
            axs[1].legend()
        logger.setLevel(level)

    def plot_model_fits(self, start_ix, count, segment="test"):
        self.energy_net.eval()
        self.timeseries_net.eval()
        level = logger.level
        logger.setLevel(logging.ERROR)
        true_loads, pred_loads = self.evaluate_over_range(
            start_ix=start_ix, count=count, segment=segment
        )
        true_loads = torch.sum(true_loads, axis=2)
        pred_loads = torch.sum(pred_loads, axis=2)
        maxes = torch.max(true_loads, dim=0)[0].reshape(1, 4)
        true_loads = true_loads / maxes
        pred_loads = pred_loads / maxes
        true_loads = true_loads.cpu()
        pred_loads = pred_loads.cpu()
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        identity = np.linspace(0, 1, 10)
        plt.suptitle("Annual Model Fits")
        axs[0, 0].scatter(true_loads[:, 0], pred_loads[:, 0], s=1, alpha=0.3)
        axs[0, 1].scatter(true_loads[:, 1], pred_loads[:, 1], s=1, alpha=0.3)
        axs[1, 0].scatter(true_loads[:, 2], pred_loads[:, 2], s=1, alpha=0.3)
        axs[1, 1].scatter(true_loads[:, 3], pred_loads[:, 3], s=1, alpha=0.3)
        axs[0, 0].plot(identity, identity, color="dodgerblue", label="Perfect Model")
        axs[0, 1].plot(identity, identity, color="dodgerblue")
        axs[1, 0].plot(identity, identity, color="dodgerblue")
        axs[1, 1].plot(identity, identity, color="dodgerblue")
        axs[0, 0].set_ylabel("Perimeter")
        axs[1, 0].set_ylabel("Core")
        axs[1, 0].set_xlabel("Heating")
        axs[1, 1].set_xlabel("Cooling")
        axs[0, 0].legend()
        fig.tight_layout()
        logger.setLevel(level)

    def plot_loss_histories(self, y_max=0.001):
        training_loss_history_array = np.array(self.training_loss_history)
        validation_loss_history_array = np.array(self.validation_loss_history)
        withheld_loss_history_array = np.array(self.withheld_loss_history)

        plt.figure(figsize=(6, 3))
        plt.plot(
            training_loss_history_array[:, 0],
            training_loss_history_array[:, 1],
            lw=0.75,
            label="Training Data Loss",
        )
        plt.plot(
            validation_loss_history_array[:, 0],
            validation_loss_history_array[:, 1],
            label="Validation Loss (in-sample EPWs)",
        )
        plt.plot(
            withheld_loss_history_array[:, 0],
            withheld_loss_history_array[:, 1],
            lw=2,
            label="Validation Loss (out-of-sample EPWs)",
        )
        # TODO: figure out better scaling for these plots
        plt.ylim([0, y_max])
        plt.legend()
        plt.show()

    def plot_true_results(self, start_ix, count, ylim):
        # results = normalize(self.results["eui"][start_ix:start_ix+count], self.eui_max, self.eui_min)
        results = self.results["eui_normalized"][start_ix : start_ix + count]
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        plt.suptitle("Simulation Results (shoebox area normalized)")
        for i in range(count):
            axs[0, 0].plot(results[i, 0], "orange", alpha=0.3)
            axs[0, 1].plot(results[i, 1], "lightblue", alpha=0.3)
            axs[1, 0].plot(results[i, 2], "orange", alpha=0.3)
            axs[1, 1].plot(results[i, 3], "lightblue", alpha=0.3)
        axs[0, 0].plot(np.mean(results[:, 0], axis=0), "orangered")
        axs[0, 1].plot(np.mean(results[:, 1], axis=0), "dodgerblue")
        axs[1, 0].plot(np.mean(results[:, 2], axis=0), "orangered")
        axs[1, 1].plot(np.mean(results[:, 3], axis=0), "dodgerblue")
        axs[0, 0].set_ylabel("Perimeter")
        axs[1, 0].set_ylabel("Core")
        axs[1, 0].set_xlabel("Heating")
        axs[1, 1].set_xlabel("Cooling")
        for i in range(2):
            for j in range(2):
                axs[i, j].set_ylim([0, ylim])
        fig.tight_layout()

    def plot_weather_vector(self, bldg_ix, param="dbt"):
        batch = self.full_storage_batch[bldg_ix : bldg_ix + 2]
        ts = self.get_batch_climate_timeseries(batch)
        ts_ix = ClimateData.series_ix[param]
        ts = ts[0, ts_ix]
        plt.figure()
        plt.title(
            f"{ClimateData.config[param]['description']} [Building {bldg_ix:05d}]"
        )
        plt.plot(ts, lw=0.5)

    def plot_params(self, start_ix, count, include_whiskers=True, title=None):
        batch = self.full_storage_batch[start_ix : start_ix + count]
        areas_norm = normalize(
            self.results["area"][start_ix : start_ix + count],
            self.area_max,
            self.area_min,
        )
        perim_areas = self.results["area_perim"][start_ix : start_ix + count]
        core_areas = self.results["area_core"][start_ix : start_ix + count]
        perim_areas_norm = normalize(
            perim_areas, self.area_perim_max, self.area_perim_min
        )
        core_areas_norm = normalize(core_areas, self.area_core_max, self.area_core_min)
        bldg_params = self.get_batch_building_vector(batch)
        names = []

        plt.figure()
        if title:
            plt.title(title)

        boxplot_params = []
        for parameter in self.schema.parameters:
            if parameter.start_ml is not None:
                vals = bldg_params[
                    :, parameter.start_ml : parameter.start_ml + parameter.len_ml
                ]
                if not isinstance(parameter, (OneHotParameter, WindowParameter)):
                    names.append(parameter.name)
                    boxplot_params.append(vals)
                elif isinstance(parameter, OneHotParameter):
                    boxplot_params.append(
                        np.argwhere(vals)[:, -1].reshape(-1, 1) / (parameter.count - 1)
                    )
                    names.append(parameter.name)
                elif isinstance(parameter, WindowParameter):
                    for i in range(vals.shape[1]):
                        vals_single = vals[:, i]
                        boxplot_params.append(vals_single.reshape(-1, 1))
                    names.append("U-Value")
                    names.append("SHGC")
                    names.append("VLT")

        boxplot_params.append(areas_norm.reshape(-1, 1))
        boxplot_params.append(perim_areas_norm.reshape(-1, 1))
        boxplot_params.append(core_areas_norm.reshape(-1, 1))
        names.append("Area")
        names.append("Area:Perim")
        names.append("Area:Core")

        offset = 1
        for i, vals in enumerate(boxplot_params):
            column_loc = np.repeat((offset), count) + np.random.normal(0, 0.02, count)
            plt.plot(column_loc, vals.flatten(), ".", alpha=0.025)
            offset = offset + 1

        if include_whiskers:
            boxplot_params = np.hstack(boxplot_params)
            plt.boxplot(boxplot_params)

        plt.xticks(ticks=list(range(1, len(names) + 1)), labels=names, rotation=90)
