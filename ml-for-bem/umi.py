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

from surrogate import Surrogate
from pyumi import UmiProject
from archetypal import UmiTemplateLibrary

from networks import EnergyCNN, MonthlyEnergyCNN
from schedules import mutate_timeseries
# from storage import download_from_bucket, upload_to_bucket
from schema import Schema, OneHotParameter, WindowParameter

from tqdm.autonotebook import tqdm

logging.basicConfig()
logger = logging.getLogger("Surrogate")
logger.setLevel(logging.INFO)

class UmiSurrogate(UmiProject):
    '''
    UMI surrogate model. 
    Currently works for a previously run umi project with set of shoeboxes.
    '''
    __slots__ = (
        "schema",
        "_surrogate", # TODO surrogate setter
        # "archetypal_template",
    )

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
        pass

    def extract_vectors_from_templates(self):
        #TODO: normalize vectors before returning
        template_batch_size = len(self.archetypal_template.BuildingTemplates) #TODO
        # templates_batch = schema.generate_empty_storage_batch(template_batch_size)
        # dict with names of templates and clipped template vector
        templates_vector_dict = {}
        for i, building_template in enumerate(self.archetypal_template.BuildingTemplates):
            vect_dict = schema.extract_from_template(building_template)
            templates_vector_dict[building_template.name] = vect_dict["template_vect"]
    
if __name__ == "__main__":
    template_path = "C:/Users/zoelh/GitRepos/ml-for-building-energy-modeling/ml-for-bem/data/template_libs/cz_libs/residential/CZ1A.json"
    template = UmiTemplateLibrary.open(template_path)
    schema = Schema()
    print(schema.extract_from_template(template.BuildingTemplates[0]))
    # umi = UmiSurrogate()