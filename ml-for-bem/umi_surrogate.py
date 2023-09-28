import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import matplotlib.pyplot as plt
import time

from pyumi import UmiProject
from pyumi.shoeboxer import ShoeBox
from pyumi.umi_project import ShoeBoxCollection

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
from archetypal.template.structure import StructureInformation
from archetypal.template.ventilation import VentilationSetting
from archetypal.template.window_setting import WindowSetting
from archetypal.template.zone_construction_set import ZoneConstructionSet
from archetypal.template.zonedefinition import ZoneDefinition

from surrogate import Surrogate, ClimateData, normalize
from weather_utils import collect_values, calc_surface_temp
from schema import Schema, OneHotParameter, WindowParameter, ShoeboxGeometryParameter, ShoeboxOrientationParameter

# from tqdm.autonotebook import tqdm

# TODO
EUI_PERIM_HEATING_MAX = 1246.034466403295  # is this for hourly or monthly?? # UNIT?
EUI_PERIM_COOLING_MAX = 845.0591803573201
EUI_CORE_HEATING_MAX = 124.12549603764043
EUI_CORE_COOLING_MAX = 110.8127478951294
AREA_MAX = 90.63  # 2000?
AREA_MIN = 5.6
PERIM_AREA_MAX = 68.84246642969192
PERIM_AREA_MIN = 0.3619999885559082
CORE_AREA_MAX = 74.86000289916991
CORE_AREA_MIN = 0.3440000057220462

J_TO_KWH = 2.7777e-7

EPLUS_DATA = [
    {
        "name": "heating",
        "eplus_name": "Zone Ideal Loads Supply Air Total Heating Energy",  # [J]
        "units": "J",
    },
    {
        "name": "cooling",
        "eplus_name": "Zone Ideal Loads Supply Air Total Cooling Energy",
        "units": "J",
    },
    {"name": "dhw", "eplus_name": "Water Use Equipment Heating Energy", "units": "J"},
    {"name": "lights", "eplus_name": "Zone Lights Electric Energy", "units": "J"},
    {
        "name": "equip",
        "eplus_name": "Zone Electric Equipment Electric Energy",
        "units": "J",
    },
    {"name": "temp", "eplus_name": "Zone Air Temperature", "units": "C"},
    {
        "name": "solargain",
        "eplus_name": "Zone Windows Total Transmitted Solar Radiation Energy",  # PERIM:Zone Windows Total Transmitted Solar Radiation Energy [J](Hourly)
        "units": "J",
    },
]

ENERGY_CSV_OUTPUTS = [
    "Date/Time",
    "PERIM:Zone People Total Heating Energy [J](Hourly)",
    "CORE:Zone People Total Heating Energy [J](Hourly)",
    "PERIM:Zone Lights Electric Energy [J](Hourly)",
    "CORE:Zone Lights Electric Energy [J](Hourly)",
    "PERIM:Zone Electric Equipment Electric Energy [J](Hourly)",
    "CORE:Zone Electric Equipment Electric Energy [J](Hourly)",
    "PERIM:Zone Windows Total Transmitted Solar Radiation Energy [J](Hourly)",
    "CORE:Zone Windows Total Transmitted Solar Radiation Energy [J](Hourly)",
    "PERIM:Zone Mean Radiant Temperature [C](Hourly)",
    "CORE:Zone Mean Radiant Temperature [C](Hourly)",
    "PERIM:Zone Mean Air Temperature [C](Hourly)",
    "PERIM:Zone Operative Temperature [C](Hourly)",
    "CORE:Zone Mean Air Temperature [C](Hourly)",
    "CORE:Zone Operative Temperature [C](Hourly)",
    "PERIM:Zone Infiltration Total Heat Loss Energy [J](Hourly)",
    "PERIM:Zone Infiltration Total Heat Gain Energy [J](Hourly)",
    "PERIM:Zone Infiltration Air Change Rate [ach](Hourly)",
    "CORE:Zone Infiltration Total Heat Loss Energy [J](Hourly)",
    "CORE:Zone Infiltration Total Heat Gain Energy [J](Hourly)",
    "CORE:Zone Infiltration Air Change Rate [ach](Hourly)",
    "PERIM:Zone Air Temperature [C](Hourly)",
    "PERIM:Zone Air Relative Humidity [%](Hourly)",
    "CORE:Zone Air Temperature [C](Hourly)",
    "CORE:Zone Air Relative Humidity [%](Hourly)",
    "PERIM IDEAL LOADS AIR:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)",
    "PERIM IDEAL LOADS AIR:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)",
    "PERIM IDEAL LOADS AIR:Zone Ideal Loads Zone Total Heating Energy [J](Hourly)",
    "CORE IDEAL LOADS AIR:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)",
    "CORE IDEAL LOADS AIR:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)",
    "CORE IDEAL LOADS AIR:Zone Ideal Loads Zone Total Heating Energy [J](Hourly)",
    "DHW PERIM:Water Use Equipment Heating Energy [J](Hourly)",
    "DHW CORE:Water Use Equipment Heating Energy [J](Hourly)",
]

logging.basicConfig()
logger = logging.getLogger("UmiSurrogate")
logger.setLevel(logging.INFO)

root_dir = Path(os.path.abspath(os.path.dirname(__file__)))
ENERGY_DIR = root_dir / "umi" / "energy"
if not os.path.exists(ENERGY_DIR):
    os.makedirs(ENERGY_DIR)
logger.info(f"Umi shoebox hourly energy will be saved in {ENERGY_DIR}")


class UmiSurrogate(UmiProject):
    """
    UMI surrogate model.
    Currently works for a previously run umi project with set of shoeboxes.
    """

    def __init__(
        self,
        schema: Schema,
        checkpoint,
        compute_loss=True,
        output_resolution=12,
        perim_offset=3.0,
        height=3.0,
        width=3.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.schema = schema
        self.compute_loss = compute_loss
        self.init_surrogate(checkpoint)
        self.output_resolution = output_resolution  # TODO
        self.shoeboxes = None
        self.surrogate = None
        self.perim_offset = perim_offset
        umi.height = height
        umi.width = width

    @classmethod
    def open(
        cls,
        umi_path,
        schema,
        checkpoint,
        compute_loss=True,
        output_resolution=12,
        name=None,
        energy_results_path=None,
        **kwargs,
    ):
        logger.info("Opening umi project. This may take a few moments...")
        umi = UmiProject.open(umi_path)
        if name:
            umi.project_name = name
        else:
            umi.project_name = os.path.split(Path(umi_path))[-1].split(".umi")[0]
        umi.__class__ = cls
        logger.info(f"Project name: {umi.project_name}")
        umi.height = umi.DEFAULT_SHOEBOX_SETTINGS["FloorToFloorHeight"]
        umi.width = umi.DEFAULT_SHOEBOX_SETTINGS["RoomWidth"]
        umi.perim_offset = umi.DEFAULT_SHOEBOX_SETTINGS["PerimeterOffset"]
        umi.output_resolution = output_resolution
        umi.schema = schema
        umi.compute_loss = compute_loss
        if energy_results_path:
            umi.set_energy_path(energy_results_path)
        # umi.shoeboxes = umi._fetch_shoeboxcollection()
        umi.init_surrogate(checkpoint, **kwargs)
        return umi

    def init_surrogate(self, checkpoint, **kwargs):
        logger.info("Setting up umi surrogate...")
        self.norm_climate_vector, self.norm_tsol_vector = self.get_climate_vector()
        logger.info(
            f"Climate vector loaded with shape {self.norm_climate_vector.shape}"
        )
        (
            self.template_vectors_dict,
            self.templates,
        ) = self.extract_vectors_from_templates()

        self.surrogate = Surrogate(
            schema=self.schema,
            checkpoint=checkpoint,
            load_training_data=False,
            **kwargs,
        )
        # if self.runtype == "val":
        #     logger.info("Validation runtype selected. Will process umi eplus outputs.")
        #     self.fetch_raw_shoebox_results()

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
        if hasattr(self, "_shoeboxdf"):
            return self._shoeboxdf
        else:
            df = pd.DataFrame.from_dict(self.sdl_common["shoebox-weights"])
            self._shoeboxdf = df.merge(
                self.gdf_world, how="left", left_on="ParentBuildingId", right_on="id"
            )
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
            for i in range(idx + 1, len(p)):
                new_p = os.path.join(new_p, p[i])
            new_paths.append(new_p)

        self.shoeboxdf["ShoeboxPath"] = new_paths
        # return self.shoeboxdf

    def _fetch_shoeboxcollection(self, start_idx=0, count=None):
        try:
            self.shoeboxes.shape
        except:
            logger.info(f"Fetching shoebox IDFs.")
            tock = time.time()
            self.shoeboxes = ShoeBoxCollection()
            # sbdf = self.shoeboxdf.iloc[start_idx:start_idx+count]
            df = self.shoeboxdf.reset_index().groupby("ShoeboxPath").first().reset_index()
            for _, row in df.iterrows():
                idf_path = row["ShoeboxPath"]
                # print(idf_path)
                # idf = IDF(idf_path)
                self.shoeboxes[idf_path] = ShoeBox(idf_path)
            tick = time.time()
            logger.info(f"Completed fetching shoebox IDFs in {round(tick-tock)} seconds.")
        return self.shoeboxes

    def fetch_shoebox_areas(self):
        # sb_lengths = (self.shoeboxdf["Core2Perimeter"] + 1) * self.perim_offset
        # areas = sb_lengths * self.width  # Ideally get width from shoebox

        # logger.info(f"Fetching areas for {df.shape[0]} shoeboxes.")

        shoeboxes = self._fetch_shoeboxcollection()
        areas = [
            shoeboxes[x].total_building_area for x in self.shoeboxdf["ShoeboxPath"]
        ]
        logger.debug(f"Got areas for all {len(areas)} shoeboxes.")
        return np.array(areas)

    def _fetch_raw_shoebox_results(self, idf_path, num_metrics, freq="Hourly"):
        # TODO: num_metrics vs list of EPLUS DATA
        array = np.zeros((2, num_metrics, 8760))  # 2 is for core, perim
        csv_path = idf_path.replace("idf", "csv")
        try:
            pandas_df = pd.read_csv(csv_path)
            pandas_df.columns = pandas_df.columns.str.strip()
            cols = pandas_df.columns.tolist()
        except Exception as e:
            print(f"Error opening energy csv: {csv_path}")
            raise e

        # TODO: if there is no water use equipment this is not added to the idf and is not in the csv output
        # Temporary fix to add zeros if there is a missing entry
        if len(cols) != len(ENERGY_CSV_OUTPUTS):
            # print(f"Missing some outputs for {csv_path}")
            for col in ENERGY_CSV_OUTPUTS:
                if col not in cols:
                    logger.info(
                        f"Could not find results for {col} in {csv_path}. Assuming no energy use (zeros)."
                    )
                    pandas_df[col] = 0
            # reorganize columns in correct order
            pandas_df = pandas_df[ENERGY_CSV_OUTPUTS]

        try:
            for z, zone in enumerate(["CORE", "PERIM"]):
                array[z, :, :] = pandas_df.filter(regex=zone).to_numpy().T
                # for m, metric in enumerate(EPLUS_DATA):
                #     if "Ideal" in metric["eplus_name"]:
                #         col_name = f'{zone} IDEAL LOADS AIR:{metric["eplus_name"]} [{metric["units"]}]({freq})'
                #     elif "Water" in metric["eplus_name"]:
                #         col_name = (
                #             f'DHW {zone}:{metric["eplus_name"]} [{metric["units"]}]({freq})'
                #         )
                #     else:
                #         col_name = (
                #             f'{zone}:{metric["eplus_name"]} [{metric["units"]}]({freq})'
                #         )
                #     # name = f'{metric["name"]}_{zone}'
                #     # res = EnergySeries.with_timeindex(
                #     #     pandas_df[col_name], units=metric["units"]
                #     # )
                #     # results[zone][name] = res
                #     array[z, m, :] = pandas_df[col_name]
        except Exception as e:
            print(f"Error reading energy csv: {csv_path}")
            raise e
        return array

    def get_storage_path(self):
        return ENERGY_DIR / f"{self.project_name}_shoeboxhourlyresults.hdf5"

    def fetch_raw_shoebox_results(self, override=False):
        hdf_path = self.get_storage_path()
        # Check if hdf file already exists:
        if not os.path.exists(hdf_path) or override:
            logger.info("Collecting energy data from shoebox outputs...")

            # groupby shoebox
            df = (
                self.shoeboxdf.reset_index()
                .groupby("ShoeboxPath")
                .first()
                .reset_index()
            )

            num_sb = self.shoeboxdf.shape[0]
            num_metrics = 16  # TODO make dynamic
            data = np.zeros((num_sb, 2, num_metrics, 8760))  # 2 is for core, perim

            res = {}
            for _, row in df.iterrows():
                i = row["index"]
                path = row["ShoeboxPath"]
                res[path] = self._fetch_raw_shoebox_results(path, num_metrics)

            for i, path in enumerate(self.shoeboxdf["ShoeboxPath"]):
                data[i] = res[path]

            with h5py.File(hdf_path, "w") as f:
                f.create_dataset(
                    self.project_name,
                    shape=data.shape,
                    data=data,
                    compression="gzip",
                    compression_opts=6,
                )

            logger.info(f"{df.shape[0]} shoebox energy results processed!")
            logger.info(f"Output shape of h5py: {data.shape}.")
        else:
            logger.info(f"Hourly output already calculated and saved under {hdf_path}.")

    def get_hourly_shoebox_results(self, start_idx=0, batch_size=None):
        """
        Returns hourly results from an external umi run.
        Format: n_shoeboxes x zones (2) x variables (19) x timesteps (8760)
        """
        hdf_path = self.get_storage_path()
        if batch_size is None:
            with h5py.File(hdf_path, "r") as f:
                data = f[self.project_name][
                    start_idx:
                ]  # this loads the whole batch into memory!
        else:
            with h5py.File(hdf_path, "r") as f:
                data = f[self.project_name][start_idx : start_idx + batch_size]
        # Zone Windows Total Transmitted Solar Radiation Energy [3]
        # Zone Ideal Loads Supply Air Total Heating Energy [12]
        # Zone Ideal Loads Supply Air Total Cooling Energy [13]
        if self.compute_loss:
            results = data[:, :, 12:14, :]
            if results.shape[-1] != self.output_resolution:
                # TODO: interpolate?
                n = int(results.shape[-1] / self.output_resolution)
                # results = np.average(
                #     results.reshape(-1, 2, 2, self.output_resolution, n), axis=4
                # )
                results = np.sum(
                    results.reshape(-1, 2, 2, self.output_resolution, n), axis=4
                )
                results = J_TO_KWH * results
                logger.info(f"Summed results to shape {results.shape}")
            return np.expand_dims(data[:, :, 3, :], 2), results
        else:
            return np.expand_dims(data[:, :, 3, :], 2), None

    def get_climate_vector(self):
        """
        One climate vector for all umi shoeboxes.
        Return np.array of shape (n_climate_params, 8760)
        """
        # if self.epw is None:
        #     self.epw()

        logger.info("Extracting climate data from umi project.")

        maxes = []
        mins = []
        for key, param in ClimateData.config.items():
            maxes.append(param["max"])
            mins.append(param["min"])
        climate_array = collect_values(self.epw)
        norm_climate_array = np.zeros(climate_array.shape)
        for i in range(climate_array.shape[0]):
            norm_climate_array[i] = normalize(climate_array[i], maxes[i], mins[i])

        # Calculate tsol air TODO replace with irradiance
        norm_tsol_array = np.zeros((4, 8760))
        for i, o in enumerate([0, 90, 180, 270]):
            norm_tsol_array[i] = calc_surface_temp(self.epw, orientation=o)
        norm_tsol_array = normalize(climate_array, maxes[-1], mins[-1])

        logger.info(f"Successfully loaded {self.epw}")
        return norm_climate_array, norm_tsol_array

    def extract_vectors_from_templates(self):
        logger.info("Collecting data from building templates...")
        # dict with names of templates and clipped template vector
        template_vectors_dict = {}
        template_names = []
        # Initialize template_lib as archetypal
        template_lib = self.init_template(self.template_lib)
        for building_template in template_lib.BuildingTemplates:
            # logger.info(
            #     f"Fetching BuildingTemplate vector data from {building_template.Name}"
            # )
            template_names.append(building_template.Name)
            vect_dict = self.schema.extract_from_template(building_template)
            template_vectors_dict[building_template.Name] = vect_dict
        # self.template_vectors_dict = template_vectors_dict
        # self.templates = template_names
        return template_vectors_dict, template_names

    def get_building_params_from_templates(self):
        """
        Return np.array of shape (n_templates, n_ml building params w/o geom)
        """
        params = np.zeros(
            (
                len(self.templates),
                self.template_vectors_dict[self.templates[0]]["template_vect"].shape[0],
            )
        )
        for i, name in enumerate(self.templates):
            params[i, :] = self.template_vectors_dict[name]["template_vect"]
        logger.info(f"Templates's geometry parameters shape={params.shape}")
        return params

    def get_schedules_from_templates(self, n_schedules=3):
        """
        Return np.array of shape (n_templates, n_scheds, 8760) - indexing template
        """
        schedules = np.zeros((len(self.templates), n_schedules, 8760))
        for i, name in enumerate(self.templates):
            schedules[i] = self.template_vectors_dict[name]["schedules_vect"]
        # self.schedules_array = schedules
        logger.info(f"Loaded Schedules Array. Shape={schedules.shape}")
        return schedules

    def get_norm_shoebox_geom_params(
        self, start_idx, count, width=3.0, height=3.0
    ):  # TODO: width setting
        """
        Return np.array of shape (n_shoeboxes, n_geom_params)
        """
        # df = np.zeros((self.shoeboxdf.shape[0], 8))
        ml_param_list = [
            p.name for p in self.schema.parameters 
            if p.in_ml 
            and isinstance(p, (ShoeboxGeometryParameter, ShoeboxOrientationParameter))
            ]
        dimension = sum(
            [
                p.shape_ml[0] for p in self.schema.parameters 
                if p.in_ml 
                and isinstance(p, (ShoeboxGeometryParameter, ShoeboxOrientationParameter))
            ]
        )
        df = np.zeros((count, dimension+3)) # add 3 for areas

        # width
        i = ml_param_list.index('width')
        df[:, i] = normalize(
            width,
            maxv=self.schema["width"].max,
            minv=self.schema["width"].min,
        )
        # height - note height in shoeboxdf is whole building height
        i = ml_param_list.index('height')
        df[:, i] = normalize(
            height,
            maxv=self.schema["height"].max,
            minv=self.schema["height"].min,
        )
        # floor_2_facade
        i = ml_param_list.index('floor_2_facade')
        df[:, i] = normalize(
            self.shoeboxdf["Floor2Fac"][start_idx : start_idx + count],
            maxv=self.schema["floor_2_facade"].max,
            minv=self.schema["floor_2_facade"].min,
        )
        # core_2_perim
        i = ml_param_list.index('core_2_perim')
        df[:, i] = normalize(
            self.shoeboxdf["Core2Perimeter"][start_idx : start_idx + count],
            maxv=self.schema["core_2_perim"].max,
            minv=self.schema["core_2_perim"].min,
        )
        # roof_2_footprint
        i = ml_param_list.index('roof_2_footprint')
        df[:, i] = normalize(
            self.shoeboxdf["Roof2FloorRatio"][start_idx : start_idx + count],
            maxv=self.schema["roof_2_footprint"].max,
            minv=self.schema["roof_2_footprint"].min,
        )
        # ground_2_footprint - TODO: is this the other way around?
        i = ml_param_list.index('ground_2_footprint')
        df[:, i] = normalize(
            self.shoeboxdf["Ground2FloorRatio"][start_idx : start_idx + count],
            maxv=self.schema["ground_2_footprint"].max,
            minv=self.schema["ground_2_footprint"].min,
        )
        # wwr
        i = ml_param_list.index('wwr')
        df[:, i] = self.shoeboxdf["WwrE"][start_idx : start_idx + count]

        # orientation TODO - help with the one hot to_ml??
        i = ml_param_list.index('orientation')
        o = self.shoeboxdf["Orientation"][start_idx : start_idx + count]
        orient_lookup = {"North": 0, "East": 1, "South": 2, "West": 3}
        orient_idxs = [orient_lookup[x] for x in o]
        orient_idxs = np.expand_dims(np.array(orient_idxs), axis=1)
        df[:, i:i+4] = self.schema["orientation"].to_ml(value=orient_idxs)

        # make tsol air array
        shoebox_norm_tsol_vector = np.zeros((count, 8760))
        for i, o in enumerate(orient_idxs):
            shoebox_norm_tsol_vector[i, :] = self.norm_tsol_vector[o, :]
        shoebox_norm_tsol_vector = np.expand_dims(shoebox_norm_tsol_vector, axis=1)
        logger.debug(f"shoebox_norm_tsol_vector shape {shoebox_norm_tsol_vector.shape}")

        # THREE AREA ONES ARE TO BE TAGGED ON AT END
        area = self.fetch_shoebox_areas()
        logger.info(f"AREA MAX: {area.max()}, MIN {area.min()}")
        area = area[start_idx : start_idx + count]
        df[:, dimension] = normalize(
            area,
            maxv=AREA_MAX,
            minv=AREA_MIN,
        )
        # perim_area = area[start_idx : start_idx + count] / (
        #     self.shoeboxdf["Core2Perimeter"][start_idx : start_idx + count] + 1
        # )
        perim_area = np.ones(area.shape) * (self.perim_offset * self.width)
        logger.info(f"PERIM AREA MAX: {perim_area.max()}, MIN {perim_area.min()}")

        i = ml_param_list.index('orientation')
        df[:, dimension+1] = normalize(
            perim_area,
            maxv=PERIM_AREA_MAX,
            minv=PERIM_AREA_MIN,
        )
        core_area = area - perim_area
        logger.info(f"CORE AREA MAX: {core_area.max()}, MIN {core_area.min()}")
        df[:, dimension+2] = normalize(
            core_area,
            maxv=CORE_AREA_MAX,
            minv=CORE_AREA_MIN,
        )

        logger.info(f"Loaded Shoebox Geometry Array. Shape={df.shape}")

        return df, shoebox_norm_tsol_vector

    def get_shoebox_template_ids(self, start_idx, count):
        """
        Return np.array of shape (n_shoeboxes) - indexing template
        """
        template_list = self.shoeboxdf["TemplateName"][start_idx : start_idx + count]
        template_idxs = [self.templates.index(x) for x in template_list]
        return np.array(template_idxs)

    def make_umi_dataset(self, start_idx, batch_size):
        """
        Make timeseries and building vectors (for ml) dynamically with each batch, starting at
        a given shoebox index.

        timeseries: Concatenates climate + orientation/irradiance-related timeseries vector

        building vector: Concatenates template geom + shoebox geom vectors
        """
        logger.debug(
            f"Constructing dataset for {batch_size} shoeboxes at index {start_idx}..."
        )

        (
            shoebox_norm_geom_params,
            shoebox_norm_tsol_vector,
        ) = self.get_norm_shoebox_geom_params(start_idx=start_idx, count=batch_size)
        logger.debug(f"shoebox_geom_params shape: {shoebox_norm_geom_params.shape}")

        logger.info("Setting up template lookup for groups...")
        shoebox_template_ids = self.get_shoebox_template_ids(
            start_idx=start_idx, count=batch_size
        )

        template_schedules = (
            self.get_schedules_from_templates()
        )  # already normalized (0-1)
        # logger.info(f"template_schedules shape: {template_schedules.shape}")

        template_norm_geom_params = self.get_building_params_from_templates()
        # logger.info(f"template_geom_params shape: {template_geom_params.shape}")

        logger.info("Constructing machine learning vectors...")
        logger.debug(f"shoebox_template_ids shape: {shoebox_template_ids.shape}")
        shoebox_norm_template_geom_params = template_norm_geom_params[
            shoebox_template_ids
        ]
        logger.debug(
            f"shoebox_template_geom_params shape: {shoebox_norm_template_geom_params.shape}"
        )

        shoebox_template_schedules = template_schedules[shoebox_template_ids]
        logger.debug(
            f"shoebox_template_schedules shape: {shoebox_template_schedules.shape}"
        )

        # CONSTRUCT BUILDING VECTOR
        building_vector = np.concatenate(
            (
                shoebox_norm_geom_params[:, :-3],
                shoebox_norm_template_geom_params,
                shoebox_norm_geom_params[:, -3:],
            ),
            axis=1,
        )
        logger.info(
            f"Building vector shape: {building_vector.shape}"
        )  # TODO: check building vector params

        # CONSTRUCT HOURLY (OR MONTHLY) RESULTS VECTOR
        # TODO: option for no results
        rad_vect, results_vector = self.get_hourly_shoebox_results(
            start_idx=start_idx, batch_size=batch_size
        )
        results_vector_normalized = []
        # TODO
        if self.compute_loss:
            # Reshape to batch_size x 4 x output_resolution
            results_vector = np.reshape(results_vector, (-1, 4, self.output_resolution))
            logger.info(
                f"RESULTS PERIM HEATING MAX: {results_vector[:, 0, :].max()}, MIN: {results_vector[:, 0, :].min()}"
            )
            logger.info(
                f"RESULTS PERIM COOLING MAX: {results_vector[:, 1, :].max()}, MIN: {results_vector[:, 1, :].min()}"
            )
            logger.info(
                f"RESULTS CORE HEATING MAX: {results_vector[:, 2, :].max()}, MIN: {results_vector[:, 0, :].min()}"
            )
            logger.info(
                f"RESULTS CORE COOLING MAX: {results_vector[:, 3, :].max()}, MIN: {results_vector[:, 1, :].min()}"
            )
            results_vector_normalized = np.zeros(results_vector.shape)

            results_vector_normalized[:, 0, :] = normalize(
                results_vector[:, 0, :], EUI_PERIM_HEATING_MAX, 0
            )
            results_vector_normalized[:, 1, :] = normalize(
                results_vector[:, 1, :], EUI_PERIM_COOLING_MAX, 0
            )
            results_vector_normalized[:, 2, :] = normalize(
                results_vector[:, 2, :], EUI_CORE_HEATING_MAX, 0
            )
            results_vector_normalized[:, 3, :] = normalize(
                results_vector[:, 3, :], EUI_CORE_COOLING_MAX, 0
            )
            logger.info(f"Results vector shape: {results_vector_normalized.shape}")

        # CONSTRUCT TIMESERIES VECTOR - batch_size x climate+rad+schedules x 8760
        timeseries_vector = np.concatenate(
            [
                np.concatenate([[self.norm_climate_vector]] * batch_size, axis=0),
                shoebox_norm_tsol_vector,
                shoebox_template_schedules,
            ],
            axis=1,
        )
        # FUTURE: use radiation values on faces
        # timeseries_vector = np.concatenate(
        #     [
        #         np.concatenate([[self.norm_climate_vector]] * batch_size, axis=0),
        #         rad_vect[:, 1, :, :],  # PERIM only for a batch x 1 x 8760
        #         shoebox_template_schedules,
        #     ],
        #     axis=1,
        # )
        logger.info(f"Timeseries vector shape: {timeseries_vector.shape}")

        return building_vector, timeseries_vector, results_vector_normalized

    def make_umi_dataloader(self, start_idx, count, dataloader_batch_size):
        building_vector, timeseries_vector, results_vector = self.make_umi_dataset(
            start_idx, count
        )

        torch.cuda.empty_cache()

        logger.debug("Building dataloaders...")
        dataset = {}
        for i in range(building_vector.shape[0]):
            # DICT ENTRIES MUST BE IN ORDER
            dataset[i] = dict(
                {
                    "building_vector": np.array(
                        [building_vector[i]] * self.output_resolution
                    ).T,
                    "timeseries_vector": timeseries_vector[i],
                    "results_vector": results_vector[i] if self.compute_loss else None,
                }
            )
        generator = torch.Generator()
        generator.manual_seed(0)

        # train, val, test = torch.utils.data.random_split(
        #     dataset, lengths=[0.8, 0.1, 0.1], generator=generator
        # )
        umi_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=dataloader_batch_size, shuffle=False
        )

        logger.info("Dataloader built.")
        return {
            "dataset": dataset,
            "dataloader": umi_dataloader,
        }

    def run_umi_batch(self, count=1000, batch_size=100):
        """
        Shoebox geometry extraction - looks at umi outputs df
        Template column to act as index
        """

        true_loads = []
        pred_loads = []
        all_losses = []

        num_sb = self.shoeboxdf.shape[0]
        start_idxs = list(range(0, num_sb, count))

        if count > num_sb:
            count = num_sb
            logger.info("Count larger than number of shoeboxes, truncating.")
        if batch_size > count:
            batch_size = count
            logger.info("Batch size larger than count, truncating.")

        logger.info(
            f"Starting umi surrogate for {num_sb} shoeboxes in {len(start_idxs)} batches."
        )
        # for it in tqdm(range(len(start_idxs))):
        for it in range(len(start_idxs)):
            idx = start_idxs[it]
            if it == len(start_idxs) - 1:
                batch_count = num_sb % count
            else:
                batch_count = count
            umi_data = self.make_umi_dataloader(idx, batch_count, batch_size)
            self._umi_dataloader = umi_data
            with torch.no_grad():
                for test_samples in umi_data["dataloader"]:
                    logger.debug("DATALOADER INPUT SHAPES")
                    logger.debug(test_samples["timeseries_vector"].shape)
                    logger.debug(test_samples["building_vector"].shape)
                    logger.debug(test_samples["results_vector"].shape)
                    projection_results = self.surrogate.project_dataloader_sample(
                        sample=test_samples, compute_loss=self.compute_loss
                    )
                    pred_loads.append(projection_results["predicted_loads"])
                    if self.compute_loss:
                        true_loads.append(projection_results["loads"])
                        all_losses.append(projection_results["loss"])

        true_loads = torch.vstack(true_loads)
        pred_loads = torch.vstack(pred_loads)
        all_losses = torch.vstack(all_losses)
        return true_loads, pred_loads, all_losses

    def surrogate_plot_params(self, start_ix, count, include_whiskers=True, title=None):
        if count > len(self._umi_dataloader["dataset"]) - start_ix:
            count = len(self._umi_dataloader["dataset"]) - start_ix
        bldg_params = np.zeros(
            (
                len(self._umi_dataloader["dataset"]),
                self._umi_dataloader["dataset"][0]["building_vector"].shape[0],
            )
        )
        for i in range(start_ix, start_ix + count):
            bldg_params[i, :] = self._umi_dataloader["dataset"][i]["building_vector"][
                :, 0
            ]

        # areas_norm = normalize(
        #     self.results["area"][start_ix : start_ix + count],
        #     self.area_max,
        #     self.area_min,
        # )
        # perim_areas = self.results["area_perim"][start_ix : start_ix + count]
        # core_areas = self.results["area_core"][start_ix : start_ix + count]
        # perim_areas_norm = normalize(
        #     perim_areas, self.area_perim_max, self.area_perim_min
        # )
        # core_areas_norm = normalize(core_areas, self.area_core_max, self.area_core_min)
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

        # boxplot_params.append(areas_norm.reshape(-1, 1))
        # boxplot_params.append(perim_areas_norm.reshape(-1, 1))
        # boxplot_params.append(core_areas_norm.reshape(-1, 1))
        boxplot_params.append(bldg_params[:, -3].reshape(-1, 1))
        boxplot_params.append(bldg_params[:, -2].reshape(-1, 1))
        boxplot_params.append(bldg_params[:, -1].reshape(-1, 1))
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

    def calculate_weighted(self):
        pass


if __name__ == "__main__":
    # template_path = "D:/Users/zoelh/GitRepos/ml-for-building-energy-modeling/ml-for-bem/data/template_libs/cz_libs/residential/CZ1A.json"
    # umi_path = "D:/Users/zoelh/GitRepos/ml-for-building-energy-modeling/umi/Sample/SampleBuildings.umi"

    umi_path = "C:/Users/zoele/Git_Repos/ml-for-building-energy-modeling/umi/SampleBuildings.umi"

    # template = UmiTemplateLibrary.open(template_path)
    # TODO: clean up loading of stuff
    schema = Schema()

    # Open and load data
    print("Opening umi project. This may take a few minutes...")
    umi = UmiSurrogate.open(umi_path=umi_path, schema=schema, checkpoint=None)
    new_p = "C:/Users/zoele/Git_Repos/ml-for-building-energy-modeling/umi/SampleBuildings/eplus"
    umi.set_energy_path(new_p)
    umi.fetch_raw_shoebox_results()

    # Set up for surrogate test
    data = umi.get_hourly_shoebox_results()
    print(type(data))
    print(data.shape)
