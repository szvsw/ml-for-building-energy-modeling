from typing import List

import logging
import os
from pathlib import Path
import numpy as np
import h5py
import torch

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
    return (data-minv)/(maxv-minv)


class ClimateData:
    # Weather data
    config = {
        "dbt": {
            "max": 50, #50.5
            "min": -35,
        },
        "rh": {
            "max": 100,
            "min": 0.0, #2.0
        },
        "atm": {
            "max": 105800.,
            "min": 75600.0,
        },
        "ghrad": {
            "max": 1200, #1154
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
        }
    }

    series_ix = {key: ix for ix,key in enumerate(config.keys())}
    series_maxes = [bounds["max"] for bounds in config.values()]
    series_mins = [bounds["min"] for bounds in config.values()]

    tsol_path = DATA_PATH / "epws" / "tsol.npy"
    climate_arr_path= DATA_PATH / "epws" / "climate_array.npy"

    __slots__ = (
        "tsol_arr",
        "climate_arr",
        "tsol_arr_norm",
        "climate_arr_norm"
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
            logging.info("Could not find Climate Array file. Will attempt to download...")
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
                self.climate_arr_norm [j, i, :] = normalize(self.climate_arr[j, i, :], ClimateData.series_maxes[i], ClimateData.series_mins[i])
            for i in range(4):
                # Handle stacked orientation time series
                self.tsol_arr_norm [j, i, :] = normalize(self.tsol_arr[j, i, :], ClimateData.config["tsol"]["max"], ClimateData.config["tsol"]["min"])
        logging.info("All climate data successfully loaded!")

class MonthlySurrogate:
    __slots__ = (
        "schema",
        "climate_data",
        "results",
        "full_input_batch",
        "default_schedules"
    )

    
    folder = DATA_PATH / "monthly_surrogate"
    os.makedirs(folder, exist_ok=True)
    default_schedules_path = folder / "default_schedules.npy"
    full_storage_batch_path = folder / "all_input_batches.hdf5"
    full_results_path = folder / "all_data_monthly.hdf5"



    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        if not os.path.exists(MonthlySurrogate.full_results_path):
            os.makedirs(DATA_PATH, exist_ok=True)
            logger.info("Full monthly data set not found.  Downloading from GCloud Bucket...")
            download_from_bucket("all_data_monthly.hdf5", MonthlySurrogate.full_results_path)
            logger.info("Done downloading dataset!")

        try:
            logging.info("Loading Default Schedules file...")
            self.default_schedules = np.load(MonthlySurrogate.default_schedules_path)
        except FileNotFoundError:
            logging.info("Could not find Default Schedules file. Will attempt to download...")
            download_from_bucket("tsol.npy", MonthlySurrogate.default_schedules_path)
            self.default_schedules = np.load(MonthlySurrogate.default_schedules_path)
        logging.info(f"Loaded Default Schedules Array.  Shape={self.default_schedules.shape}")

        logger.info("Loading the full dataset into main RAM...")
        self.results = {}
        with h5py.File(MonthlySurrogate.full_results_path, 'r') as f:
            self.results["area"] = f["area"][...]
            self.results[ "total_heating"  ]= f["total_heating"][...] # this loads the whole batch into memory!
            self.results[ "total_cooling"  ]= f["total_cooling"][...] # this loads the whole batch into memory!
            self.results[ "window_u"  ]= f["window_u"][...]
            self.results[ "facade_hcp"  ]= f["facade_hcp"][...]
            self.results[ "roof_hcp"  ]= f["roof_hcp"][...]
            self.results[ "monthly"  ]= f["monthly"][...] # this loads the whole batch into memory!
            self.full_input_batch = f["storage_batch"][...]
        
        logger.info("Finished loading the full dataset.")
        logger.info("Loading climate data...")
        self.climate_data = ClimateData()
        logger.info("Finished loading climate data.")
        
    
    def get_batch_climate_timeseries(self, batch):
        orientations = self.schema["orientation"].extract_storage_values_batch(batch).astype(int)
        epw_idxs = self.schema["base_epw"].extract_storage_values_batch(batch).astype(int)
        epw_data = self.climate_data.climate_arr_norm[epw_idxs].squeeze()
        tsol_data = self.climate_data.tsol_arr_norm[epw_idxs.flatten(),  orientations.flatten()].reshape(-1, 1, HOURS_IN_YEAR)
        climate_timeseries_vector = np.concatenate((epw_data,tsol_data),axis=1)
        return climate_timeseries_vector
    
    # TODO: use new uvalue/facade etc
    def get_building_data(self, batch):
        bldg_params, timeseries_ops = self.schema.to_ml(batch)
        seeds = self.schema["schedules_seed"].extract_storage_values_batch(batch)
        schedules = MonthlySurrogate.default_schedules
        # mutate_timeseries()

        return bldg_params, None#timeseries


    