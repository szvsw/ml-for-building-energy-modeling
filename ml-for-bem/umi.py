import logging
import os
from typing import List
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from surrogate import Surrogate, ClimateData, normalize
from pyumi import UmiProject
from archetypal import UmiTemplateLibrary
from weather_utils import collect_values

from networks import EnergyCNN, MonthlyEnergyCNN
from schedules import mutate_timeseries
# from storage import download_from_bucket, upload_to_bucket
from schema import Schema, OneHotParameter, WindowParameter

from tqdm.autonotebook import tqdm

logging.basicConfig()
logger = logging.getLogger("Surrogate")
logger.setLevel(logging.INFO)

class UmiSurrogate():
    '''
    UMI surrogate model. 
    Currently works for a previously run umi project with set of shoeboxes.
    '''

    def __init__(
        self, 
        umi_project: UmiProject,
        schema: Schema,
        surrogate: Surrogate,
        *args, **kwargs
        ):
        # super().__init__(*args, **kwargs)
        
        self.schema = schema
        self.umi_project = umi_project
        self._surrogate = surrogate

    def umi_batch(self):
        """
        Shoebox geometry extraction - looks at umi outputs df
        Template column to act as index
        """
        pass

    def extract_climate_vector(self):
        """
        One climate vector for all umi shoeboxes
        """
        # if self.epw is None:
        #     self.epw()
        maxes = []
        mins = []
        for key, param in ClimateData.config.items():
            maxes.append(param['max'])
            mins.append(param['min'])
        climate_array = collect_values(self.umi_project.epw)
        norm_climate_array = np.zeros(climate_array.shape)
        for i in range(climate_array.shape[0]):
            norm_climate_array[i] = normalize(climate_array[i], maxes[i], mins[i])
        self.climate_vector = norm_climate_array

    def extract_vectors_from_templates(self):
        logger.info("Collecting data from building templates...")
        # dict with names of templates and clipped template vector
        template_vectors_dict = {}
        for building_template in self.umi_project.template_lib.BuildingTemplates:
            logger.info(f"Fetching BuildingTemplate vector data from {building_template.Name}")
            vect_dict = schema.extract_from_template(building_template)
            template_vectors_dict[building_template.Name] = vect_dict
        self.template_vectors = template_vectors_dict
    
if __name__ == "__main__":
    template_path = "D:/Users/zoelh/GitRepos/ml-for-building-energy-modeling/ml-for-bem/data/template_libs/cz_libs/residential/CZ1A.json"
    umi_path = "D:/Users/zoelh/Dropbox (MIT)/Downgrades/2 Oshkosh/1 Baseline_mid/230417Oshkosh_mid.umi"

    # template = UmiTemplateLibrary.open(template_path)
    schema = Schema()
    surrogate = Surrogate(schema=schema)
    print("Opening umi project. This may take a few minutes...")
    umi_project = UmiProject.open(umi_path)
    umi = UmiSurrogate(schema=schema, umi_project=umi_project, surrogate=surrogate)
    print(umi.umi_project.epw)
    umi.extract_climate_vector()
    print(umi.climate_vector.shape)
    # umi.extract_vectors_from_templates()
    # print(umi.template_vectors)
