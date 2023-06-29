import logging
import os
from typing import List
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from surrogate import Surrogate, ClimateData, normalize
from pyumi import UmiProject
from archetypal import UmiTemplateLibrary
from archetypal.idfclass.idf import IDF
from archetypal.template.building_template import BuildingTemplate
from archetypal.template.conditioning import ZoneConditioning
from archetypal.template.constructions.opaque_construction import OpaqueConstruction
from archetypal.template.constructions.window_construction import WindowConstruction
from archetypal.template.dhw import DomesticHotWaterSetting
from archetypal.template.load import ZoneLoad
from archetypal.template.materials.gas_layer import GasLayer
from archetypal.template.materials.gas_material import GasMaterial
from archetypal.template.materials.glazing_material import GlazingMaterial
from archetypal.template.materials.material_layer import MaterialLayer
from archetypal.template.materials.opaque_material import OpaqueMaterial
from archetypal.template.schedule import (
    DaySchedule,
    UmiSchedule,
    WeekSchedule,
    YearSchedule,
    YearSchedulePart,
)
from archetypal.template.structure import MassRatio, StructureInformation
from archetypal.template.umi_base import UmiBase, UniqueName
from archetypal.template.ventilation import VentilationSetting
from archetypal.template.window_setting import WindowSetting
from archetypal.template.zone_construction_set import ZoneConstructionSet
from archetypal.template.zonedefinition import ZoneDefinition

from weather_utils import collect_values

from networks import EnergyCNN, MonthlyEnergyCNN
from schedules import mutate_timeseries
# from storage import download_from_bucket, upload_to_bucket
from schema import Schema, OneHotParameter, WindowParameter

from tqdm.autonotebook import tqdm

from energy_pandas import EnergyDataFrame, EnergySeries

SHOEBOX_THRESHOLD = 1000

EPLUS_DATA = [
    {"name": "heating",
    "eplus_name": "Zone Ideal Loads Supply Air Total Heating Energy", # [J]
    "units": "J"
    },
    {"name": "cooling",
    "eplus_name": "Zone Ideal Loads Supply Air Total Cooling Energy",
    "units": "J"
    },
    {"name": "dhw",
    "eplus_name": "Water Use Equipment Heating Energy",
    "units": "J"
    },
    {"name": "lights",
    "eplus_name": "Zone Lights Electric Energy",
    "units": "J"
    },
    {"name": "equip",
    "eplus_name": "Zone Electric Equipment Electric Energy",
    "units": "J"
    },
    {"name": "temp",
    "eplus_name": "Zone Air Temperature",
    "units": "C"
    },
    {"name": "solargain",
    "eplus_name": "Zone Windows Total Transmitted Solar Radiation Energy", # PERIM:Zone Windows Total Transmitted Solar Radiation Energy [J](Hourly)
    "units": "J"
    },
]

logging.basicConfig()
logger = logging.getLogger("UmiSurrogate")
logger.setLevel(logging.INFO)

root_dir = Path(os.path.abspath(os.path.dirname(__file__)))
ENERGY_DIR = root_dir / "umi" / "energy"
if not os.path.exists(ENERGY_DIR):
   os.makedirs(ENERGY_DIR)

class UmiSurrogate(UmiProject):
    '''
    UMI surrogate model. 
    Currently works for a previously run umi project with set of shoeboxes.
    '''

    def __init__(
        self, 
        umi,
        schema: Schema,
        surrogate: Surrogate,
        *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        self.schema = schema
        # self.umi_project = umi_project
        self._surrogate = surrogate
        # self.open(umi_path)

    @classmethod
    def open(cls, umi_path, schema, surrogate):
        umi = UmiProject.open(umi_path)
        umi.__class__ = cls
        umi.schema = schema
        umi._surrogate = surrogate
        return umi
    
    def init_template(self, datastore):
        t = UmiTemplateLibrary("lib")
        t.GasMaterials = [
            GasMaterial.from_dict(store, allow_duplicates=False)
            for store in datastore["GasMaterials"]
        ]
        t.GlazingMaterials = [
            GlazingMaterial.from_dict(store, allow_duplicates=False)
            for store in datastore["GlazingMaterials"]
        ]
        t.OpaqueMaterials = [
            OpaqueMaterial.from_dict(store, allow_duplicates=False)
            for store in datastore["OpaqueMaterials"]
        ]
        t.OpaqueConstructions = [
            OpaqueConstruction.from_dict(
                store,
                materials={
                    a.id: a
                    for a in (t.GasMaterials + t.GlazingMaterials + t.OpaqueMaterials)
                },
                allow_duplicates=True,
            )
            for store in datastore["OpaqueConstructions"]
        ]
        t.WindowConstructions = [
            WindowConstruction.from_dict(
                store,
                materials={a.id: a for a in (t.GasMaterials + t.GlazingMaterials)},
                allow_duplicates=True,
            )
            for store in datastore["WindowConstructions"]
        ]
        t.StructureInformations = [
            StructureInformation.from_dict(
                store,
                materials={a.id: a for a in t.OpaqueMaterials},
                allow_duplicates=True,
            )
            for store in datastore["StructureDefinitions"]
        ]
        t.DaySchedules = [
            DaySchedule.from_dict(store, allow_duplicates=True)
            for store in datastore["DaySchedules"]
        ]
        t.WeekSchedules = [
            WeekSchedule.from_dict(
                store,
                day_schedules={a.id: a for a in t.DaySchedules},
                allow_duplicates=True,
            )
            for store in datastore["WeekSchedules"]
        ]
        t.YearSchedules = [
            YearSchedule.from_dict(
                store,
                week_schedules={a.id: a for a in t.WeekSchedules},
                allow_duplicates=True,
            )
            for store in datastore["YearSchedules"]
        ]
        t.DomesticHotWaterSettings = [
            DomesticHotWaterSetting.from_dict(
                store,
                schedules={a.id: a for a in t.YearSchedules},
                allow_duplicates=True,
            )
            for store in datastore["DomesticHotWaterSettings"]
        ]
        t.VentilationSettings = [
            VentilationSetting.from_dict(
                store,
                schedules={a.id: a for a in t.YearSchedules},
                allow_duplicates=True,
            )
            for store in datastore["VentilationSettings"]
        ]
        t.ZoneConditionings = [
            ZoneConditioning.from_dict(
                store,
                schedules={a.id: a for a in t.YearSchedules},
                allow_duplicates=True,
            )
            for store in datastore["ZoneConditionings"]
        ]
        t.ZoneConstructionSets = [
            ZoneConstructionSet.from_dict(
                store,
                opaque_constructions={a.id: a for a in t.OpaqueConstructions},
                allow_duplicates=True,
            )
            for store in datastore["ZoneConstructionSets"]
        ]
        t.ZoneLoads = [
            ZoneLoad.from_dict(
                store,
                schedules={a.id: a for a in t.YearSchedules},
                allow_duplicates=True,
            )
            for store in datastore["ZoneLoads"]
        ]
        t.ZoneDefinitions = [
            ZoneDefinition.from_dict(
                store,
                zone_conditionings={a.id: a for a in t.ZoneConditionings},
                zone_construction_sets={a.id: a for a in t.ZoneConstructionSets},
                domestic_hot_water_settings={
                    a.id: a for a in t.DomesticHotWaterSettings
                },
                opaque_constructions={a.id: a for a in t.OpaqueConstructions},
                zone_loads={a.id: a for a in t.ZoneLoads},
                ventilation_settings={a.id: a for a in t.VentilationSettings},
                allow_duplicates=True,
            )
            for store in datastore["Zones"]
        ]
        t.WindowSettings = [
            WindowSetting.from_ref(
                store["$ref"],
                datastore["BuildingTemplates"],
                schedules={a.id: a for a in t.YearSchedules},
                window_constructions={a.id: a for a in t.WindowConstructions},
            )
            if "$ref" in store
            else WindowSetting.from_dict(
                store,
                schedules={a.id: a for a in t.YearSchedules},
                window_constructions={a.id: a for a in t.WindowConstructions},
                allow_duplicates=True,
            )
            for store in datastore["WindowSettings"]
        ]
        t.BuildingTemplates = [
            BuildingTemplate.from_dict(
                store,
                zone_definitions={a.id: a for a in t.ZoneDefinitions},
                structure_definitions={a.id: a for a in t.StructureInformations},
                window_settings={a.id: a for a in t.WindowSettings},
                schedules={a.id: a for a in t.YearSchedules},
                window_constructions={a.id: a for a in t.WindowConstructions},
                allow_duplicates=True,
            )
            for store in datastore["BuildingTemplates"]
        ]
        return t

    @property
    def shoeboxdf(self):
        if hasattr(self, '_shoeboxdf'):
            return self._shoeboxdf
        else:
            self._shoeboxdf = pd.DataFrame.from_dict(self.sdl_common["shoebox-weights"])
            return self._shoeboxdf
    
    def set_energy_path(self, new_path):
        old_paths = self.shoeboxdf["ShoeboxPath"]
        new_paths = []

        new_path = os.path.normpath(new_path)
        to_match = new_path.split(os.sep)[-1]
        idx = old_paths[0].split("\\").index(to_match)
        for p in old_paths:
            p = p.split("\\")
            new_p = new_path
            for i in range(idx+1, len(p)):
                new_p = os.path.join(new_p, p[i])
            new_paths.append(new_p)
        
        self.shoeboxdf["ShoeboxPath"] = new_paths
        return self.shoeboxdf
    
    def _fetch_raw_shoebox_results(self, idf_path, freq="Hourly"):
        csv_path = idf_path.replace("idf", "csv")
        pandas_df = pd.read_csv(csv_path)
        pandas_df.columns = pandas_df.columns.str.strip()
        results = {
            "CORE": EnergyDataFrame(data=[], name=idf_path),
            "PERIM": EnergyDataFrame(data=[], name=idf_path)
            }
        for zone in ["CORE", "PERIM"]:
            for metric in EPLUS_DATA:
                if "Ideal" in metric["eplus_name"]:
                    col_name = f'{zone} IDEAL LOADS AIR:{metric["eplus_name"]} [{metric["units"]}]({freq})'
                elif "Water" in metric["eplus_name"]:
                    col_name = f'DHW {zone}:{metric["eplus_name"]} [{metric["units"]}]({freq})'
                else:
                    col_name = f'{zone}:{metric["eplus_name"]} [{metric["units"]}]({freq})'
                name = f'{metric["name"]}_{zone}'
                res = EnergySeries.with_timeindex(pandas_df[col_name], units=metric["units"])
                results[zone][name] = res
        return results
    
    def fetch_raw_shoebox_results(self):
        # TODO: should we refrence shoeboxes by an id instead?
        logger.info("Collecting energy data from shoebox outputs...")

        num_sb = self.shoeboxdf.shape[0]
        if num_sb > SHOEBOX_THRESHOLD:
            logger.warning("Too many shoeboxes to save data locally...")
        
        self._raw_shoebox_results = []

        # groupby shoebox
        df = self.shoeboxdf.groupby('ShoeboxPath').first().reset_index()

        for i, row in df.iterrows():
            path_pcs = row['ShoeboxPath'].replace('.idf', '').split("\\")
            name = "eplus"
            for i in range(path_pcs.index("eplus")+1, len(path_pcs)):
                name += "_" + str(path_pcs[i])
            hdf_path = ENERGY_DIR / f"{name}.hdf5"
            res = self._fetch_raw_shoebox_results(row['ShoeboxPath'])
            # TODO: where do we want to store these??? Save as a numpy array?
            for zone in ["CORE", "PERIM"]:
                res[zone].to_hdf(hdf_path, key=zone)
            if num_sb <= SHOEBOX_THRESHOLD:
                self._raw_shoebox_results.append({
                    "ShoeboxPath": row['ShoeboxPath'],
                    "data": res,
                    })
        logger.info(f"{df.shape[0]} shoebox energy results processed!")
            

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
        climate_array = collect_values(self.epw)
        norm_climate_array = np.zeros(climate_array.shape)
        for i in range(climate_array.shape[0]):
            norm_climate_array[i] = normalize(climate_array[i], maxes[i], mins[i])
        self.climate_vector = norm_climate_array

    def extract_vectors_from_templates(self):
        logger.info("Collecting data from building templates...")
        # dict with names of templates and clipped template vector
        template_vectors_dict = {}
        # Initialize template_lib as archetypal
        template_lib = self.init_template(self.template_lib)
        for building_template in template_lib.BuildingTemplates:
            logger.info(f"Fetching BuildingTemplate vector data from {building_template.Name}")
            vect_dict = self.schema.extract_from_template(building_template)
            template_vectors_dict[building_template.Name] = vect_dict
        self.template_vectors = template_vectors_dict
    
if __name__ == "__main__":
    template_path = "D:/Users/zoelh/GitRepos/ml-for-building-energy-modeling/ml-for-bem/data/template_libs/cz_libs/residential/CZ1A.json"
    umi_path = "D:/Users/zoelh/GitRepos/ml-for-building-energy-modeling/umi/Sample/SampleBuildings.umi"

    # template = UmiTemplateLibrary.open(template_path)
    # TODO: clean up loading of stuff
    schema = Schema()
    surrogate = Surrogate(schema=schema) #, runtype="umi"
    print("Opening umi project. This may take a few minutes...")
    # umi_project = UmiProject.open(umi_path)
    umi = UmiSurrogate.open(umi_path=umi_path, schema=schema, surrogate=surrogate)
    print(umi.epw)
    umi.extract_climate_vector()
    print(umi.climate_vector.shape)
    umi.extract_vectors_from_templates()
    # print(umi.template_vectors)
    print(umi.sdl_common)
    # print(umi.shoeboxdf)
