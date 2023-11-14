import json
import logging
import math
import tempfile
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Tuple, List, Literal, Union
from zipfile import ZipFile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from archetypal import UmiTemplateLibrary
from archetypal.template.building_template import BuildingTemplate
from archetypal.template.conditioning import ZoneConditioning
from archetypal.template.constructions.opaque_construction import OpaqueConstruction
from archetypal.template.constructions.window_construction import WindowConstruction
from archetypal.template.dhw import DomesticHotWaterSetting
from archetypal.template.load import ZoneLoad
from archetypal.template.materials.gas_material import GasMaterial
from archetypal.template.materials.glazing_material import GlazingMaterial
from archetypal.template.materials.opaque_material import OpaqueMaterial
from archetypal.template.schedule import DaySchedule, WeekSchedule, YearSchedule
from archetypal.template.structure import StructureInformation
from archetypal.template.ventilation import VentilationSetting
from archetypal.template.window_setting import WindowSetting
from archetypal.template.zone_construction_set import ZoneConstructionSet
from archetypal.template.zonedefinition import ZoneDefinition
from geopandas import GeoDataFrame
from ladybug.epw import EPW
from pyproj import CRS

from shoeboxer.builder import template_dict
from umi.trace import Sky, Tracer
from utils.constants import *
from utils.schedules import get_schedules
from weather.weather import extract

logging.basicConfig()
logger = logging.getLogger("UMI")
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)


def init_template(datastore):
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
            domestic_hot_water_settings={a.id: a for a in t.DomesticHotWaterSettings},
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


# TODO: move to constants?
floor_names = ["bottom", "middle", "top", "exclusive"]
exposure_ratios_roof = [0.0, 0.0, 1.0, 1.0]
exposure_ratios_ground = [1.0, 0.0, 0.0, 1.0]


class UBEM:
    def __init__(
        self,
        gdf: GeoDataFrame,
        height_col: str,
        wwr_col: str,
        id_col: str,
        template_name_col: str,
        epw: EPW,
        template_lib: Union[Tuple[pd.DataFrame, np.ndarray], UmiTemplateLibrary],
        sensor_spacing=3,  # sensor width is used in raytracing
        shoebox_width=3,  # -1 is dynamic based off of edge length TODO: support for list shoebox_width should come from a gdf column, similar to wwr
        floor_to_floor_height=4,  # TODO: support for list floor_to_floor_height dynamic should come from a gdf column, similar to wwr
        perim_offset=PERIM_OFFSET,
        shoebox_gen_type: Literal[
            "cardinal_unshaded",
            "edge_unshaded",
            "area",
            "raytrace",
        ] = "edge_unshaded",
    ):
        # TODO:
        # WHY ARE THE NUMBER OF SBS DIFFERENT when opening an UMI vs from gdf only

        sb_calculation_options = [
            "cardinal_unshaded",
            "edge_unshaded",
            "raytrace",
            "area",
        ]
        assert (
            shoebox_gen_type in sb_calculation_options
        ), f"Shoebox generation type must be one of {sb_calculation_options}"

        # Store column metadata for building_gdf
        self.height_col = height_col
        self.id_col = id_col
        self.wwr_col = wwr_col
        self.template_name_col = template_name_col
        self.template_idx_col = "template_idx"

        # store objects
        self.building_gdf = gdf
        self.epw = epw

        # set geometric variables
        self.sensor_spacing = sensor_spacing
        self.shoebox_width = shoebox_width
        self.floor_to_floor_height = floor_to_floor_height
        self.perim_offset = perim_offset

        start_time = time.time()
        if isinstance(template_lib, UmiTemplateLibrary):
            assert len([t.Name for t in template_lib.BuildingTemplates]) == len(
                set([t.Name for t in template_lib.BuildingTemplates])
            ), "Duplicate names in template library!  Aborting."
            (
                self.template_features_df,
                self.schedules_array,
            ) = self.prepare_archetypal_features(template_lib)
        else:
            assert isinstance(
                template_lib, tuple
            ), f"template_lib must be a tuple of (features:pd.DataFrame, schedules:np.ndarray), but is {type(template_lib)}"
            assert (
                len(template_lib) == 2
            ), f"template_lib must be a tuple of (features:pd.DataFrame, schedules:np.ndarray), but is {len(template_lib)} items long instead of 2."
            assert isinstance(
                template_lib[0], pd.DataFrame
            ), f"template_lib must be a tuple of (features: pd.DataFrame, schedules:np.ndarray), but the features are {type(template_lib[0])}"
            assert isinstance(
                template_lib[1], np.ndarray
            ), f"template_lib must be a tuple of (features:pd.DataFrame, schedules:np.ndarray), but the schedules are {type(template_lib[1])}"
            features, schedules = template_lib[0], template_lib[1]
            if self.template_idx_col not in features.columns:
                features[self.template_idx_col] = range(len(features))
            if self.template_name_col not in features.columns:
                features[self.template_name_col] = features.index
            self.template_features_df = features
            self.schedules_array = schedules

        # Add in template_idx for feature array
        self.building_gdf[self.template_idx_col] = list(
            self.template_features_df.loc[self.building_gdf[self.template_name_col]][
                self.template_idx_col
            ]
        )
        self.epw_array = self.prepare_epw_features(self.epw)
        self.gis_features_df = self.prepare_gis_features()
        self.shoeboxes_df = self.prepare_shoeboxes(shoebox_gen_type)

        logger.info(f"Processed UMI in {time.time() - start_time:,.2f} seconds")

    def prepare_gis_features(self) -> pd.DataFrame:
        """
        Prepares geometric features from gis data.  These are the geometric features which
        are always constant for entire buildings, e.g. heights, core_2_perim, floor_2_facade,
        floor_count, footprint_area, etc, but does not include e.g. roof_2_footprint,
        ground_2_footprint, or shading

        Returns:
            geometric_features_df (pd.DataFrame): df with geometric features
        """
        # TODO: courtyards
        self.building_gdf["cores"] = self.building_gdf["geometry"].buffer(
            -1 * self.perim_offset
        )
        ids = self.building_gdf[self.id_col]
        if len(ids) != len(ids.unique()):
            logger.warning("Duplicate ids in building gdf! Overwriting with new ids.")
            self.building_gdf[self.id_col] = range(len(ids))
            ids = self.building_gdf[self.id_col]

        heights = self.building_gdf[self.height_col]
        wwrs = self.building_gdf[self.wwr_col]
        template_names = self.building_gdf[self.template_name_col]
        footprint_areas = self.building_gdf.geometry.area
        perimeter_length = self.building_gdf.geometry.length
        core_areas = self.building_gdf.cores.area
        if core_areas.min() > 0:
            logger.warning("Negative core areas found.  Setting to zero.")
            core_areas = np.where(core_areas < 0, 0, core_areas)

        assert (
            core_areas.min() >= 0
        ), "Core areas must be greater than zero; if not, they should ."
        perim_areas = footprint_areas - core_areas
        facade_area = self.floor_to_floor_height * perimeter_length
        perim_area_to_facade_area = perim_areas / facade_area
        core_area_to_perimeter_area = core_areas / perim_areas
        floor_count = round(heights / self.floor_to_floor_height)
        # TODO: reset floor_to_floor height so there is a whole number of floors?

        geometric_features_df = pd.DataFrame()
        geometric_features_df["building_id"] = ids
        geometric_features_df["building_idx"] = range(len(geometric_features_df))
        geometric_features_df["wwr"] = wwrs
        geometric_features_df["bldg_height"] = heights
        geometric_features_df["bldg_footprint_area"] = footprint_areas
        geometric_features_df["bldg_core_area"] = core_areas
        geometric_features_df["bldg_perim_area"] = perim_areas
        geometric_features_df["bldg_facade_area"] = facade_area
        geometric_features_df[
            "bldg_core_area_to_perim_area"
        ] = core_area_to_perimeter_area
        geometric_features_df[
            "bldg_perim_area_to_facade_area"
        ] = perim_area_to_facade_area
        geometric_features_df["floor_count"] = floor_count
        geometric_features_df["template_name"] = template_names

        # TODO: this could be a separate melted df
        # which is then used for merging rather than pick_weights
        # Compute the floor weight for each building
        # If there is more than 1 floor, then we use the four shoeboxes from the bottom floor exactly once
        # and the four shoeboxes from the top floor exactly once
        # If there are more than 2 floors, then we use the four shoeboxes from the middle floors floor_count - 2 times
        # Otherwise if there is exactly 1 floor, then we use the Exclusive shoeboxes for each direction exactly once
        geometric_features_df["bottom_weight"] = (
            geometric_features_df["floor_count"] > 1
        ).astype(int)
        geometric_features_df["top_weight"] = (
            geometric_features_df["floor_count"] > 1
        ).astype(int)
        geometric_features_df["middle_weight"] = (
            geometric_features_df["floor_count"] > 2
        ).astype(int) * (geometric_features_df["floor_count"] - 2)

        geometric_features_df["exclusive_weight"] = (
            geometric_features_df["floor_count"] == 1
        ).astype(int)
        floor_counts = geometric_features_df["floor_count"].values.reshape(-1, 1)
        for name in floor_names:
            assert f"{name}_weight" in geometric_features_df.columns

        # normalize floor_totals
        floor_weight_names = [f"{floor_name}_weight" for floor_name in floor_names]
        geometric_features_df[floor_weight_names] = (
            geometric_features_df[floor_weight_names].values / floor_counts
        ).clip(0, 1)

        return geometric_features_df

    def prepare_archetypal_features(
        self,
        template_lib: UmiTemplateLibrary,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fetches data from archetypal building templates for shoeboxes.

        Args:
            UmiTemplateLibrary: archetypal template library

        Returns:
            template_df (pd.DataFrame): a pandas df of template features that are used in the template_dict of the shoebox builder (and surrogate)
            schedules (np.ndarray):  a numpy array of schedule data for each building template (n_templates, n_used_templates (3), 8760)
        """

        # check that there are no duplicate names in the template library
        names = [t.Name for t in template_lib.BuildingTemplates]
        assert len(names) == len(
            set(names)
        ), "Duplicate names in template library!  Aborting."

        # Check that all the templates in the gdf are in the template library
        needed_templates = list(self.building_gdf[self.template_name_col].unique())
        assert all(
            [temp_name in names for temp_name in needed_templates]
        ), f"At least one of the templates in the GDF template assignment column {self.template_name_col} is not in the template library.  Aborting!"

        # get the template data
        template_data_dict = {}
        template_names = []
        for i, building_template in enumerate(template_lib.BuildingTemplates):
            # Skip unneeded templates
            if building_template.Name not in needed_templates:
                logger.debug(
                    f"Skipping {building_template.Name} since it is not used in the GDF."
                )
                continue

            data = self.dict_from_buildingtemplate(building_template)
            data[self.template_name_col] = building_template.Name
            data[self.template_idx_col] = i
            template_data_dict[building_template.Name] = data

        template_names = [x.Name for x in template_lib.BuildingTemplates]
        n_templates = len(template_lib.BuildingTemplates)
        schedules = np.zeros((n_templates, len(SCHEDULE_PATHS), 8760))
        for name, d in template_data_dict.items():
            i = template_names.index(name)
            schedules[i] = d.pop("schedules")
        # TODO: many of these are not returning with correct dtypes
        template_features = pd.DataFrame.from_dict(template_data_dict).T

        # TODO: can remove later
        template_features[
            "cop_heating"
        ] = building_template.Perimeter.Conditioning.HeatingCoeffOfPerf
        template_features[
            "cop_cooling"
        ] = building_template.Perimeter.Conditioning.CoolingCoeffOfPerf

        return template_features, schedules

    @classmethod
    def dict_from_buildingtemplate(cls, building_template: BuildingTemplate):
        logger.debug(
            f"Fetching BuildingTemplate vector data from {building_template.Name}"
        )
        # 1. Get the schedules
        scheds = get_schedules(
            building_template, zones=["Perimeter"], paths=SCHEDULE_PATHS
        )
        # 2. Get the construction factors
        for name, constr in building_template.Perimeter.Constructions:
            if name == "Facade":
                wall_r = constr.r_value
                FacadeMass = cls.sort_tmass(constr.heat_capacity_per_unit_wall_area)
                logger.debug(
                    f"Found facade with r_value {round(wall_r, 2)} and tmass bin {FacadeMass}"
                )
            if name == "Roof":
                roof_r = constr.r_value
                RoofMass = cls.sort_tmass(constr.heat_capacity_per_unit_wall_area)
                logger.debug(
                    f"Found roof with r_value {round(roof_r, 2)} and tmass bin {RoofMass}"
                )
            if name == "Ground":
                slab_r = constr.r_value
                logger.debug(f"Found slab with r_value {round(slab_r, 2)}")

        # 3. Get window parameters
        window_u = 1 / building_template.Windows.Construction.r_value
        try:
            shgc = 1 / building_template.Windows.Construction.shgc()
        except:
            logger.debug("Using internal shgc calculation.")
            tsol = (
                1
                / building_template.Windows.Construction.Layers[
                    0
                ].Material.SolarTransmittance
            )
            shgc = cls.single_pane_shgc_estimation(tsol, window_u)

        if shgc > 1:
            logger.warning("SHGC over 1, clipping.")
            shgc = 1.0

        # 4. Get the ventilation factors
        # TODO: how to deal with things that are named weird
        vent_sched_name = building_template.Perimeter.Conditioning.MechVentSchedule.Name
        if building_template.Perimeter.Conditioning.MechVentSchedule == 0:
            vent_mode = MechVentMode.Off
        elif "ALWAYSOFF" in vent_sched_name.upper():
            vent_mode = MechVentMode.AllOff
        elif "ALWAYSON" in vent_sched_name.upper():
            vent_mode = MechVentMode.AllOn
        elif "OCC" in vent_sched_name.upper():
            vent_mode = MechVentMode.OccupancySchedule
        else:
            logger.warning(
                "Mechanical ventilation response for schedule {vent_sched_name} is not supported. Defaulting to occupancy."
            )
            vent_mode = MechVentMode.OccupancySchedule

        recovery_type = building_template.Perimeter.Conditioning.HeatRecoveryType.name
        if recovery_type == "NONE":
            recovery_type = "NoHRV"
        assert recovery_type in (x.name for x in HRV)

        econ_type = building_template.Perimeter.Conditioning.EconomizerType.name
        if econ_type not in (x.name for x in Econ):
            logger.warning(
                f"Economizer type {econ_type} is not supported. Defaulting to DifferentialEnthalpy"
            )
            econ_type = "DifferentialEnthalpy"
        assert econ_type in (x.name for x in Econ)

        logger.debug(f"Thermal mass roof {RoofMass}, walls {FacadeMass}.")

        td = template_dict(
            schedules=scheds,
            PeopleDensity=building_template.Perimeter.Loads.PeopleDensity,
            LightingPowerDensity=building_template.Perimeter.Loads.LightingPowerDensity,
            EquipmentPowerDensity=building_template.Perimeter.Loads.EquipmentPowerDensity,
            Infiltration=building_template.Perimeter.Ventilation.Infiltration,
            VentilationPerArea=building_template.Perimeter.Conditioning.MinFreshAirPerArea,
            VentilationPerPerson=building_template.Perimeter.Conditioning.MinFreshAirPerPerson,
            VentilationMode=vent_mode,
            HeatingSetpoint=building_template.Perimeter.Conditioning.HeatingSetpoint,
            CoolingSetpoint=building_template.Perimeter.Conditioning.CoolingSetpoint,
            RecoverySettings=getattr(HRV, recovery_type),
            EconomizerSettings=getattr(Econ, econ_type),
            FacadeRValue=wall_r,
            FacadeMass=FacadeMass,
            RoofRValue=roof_r,
            RoofMass=RoofMass,
            SlabRValue=slab_r,
            WindowShgc=shgc,
            WindowUValue=window_u,
        )
        return td

    @classmethod
    def sort_tmass(cls, val):
        """
        Sorts thermal mass into categories based on capacity

        Args:
            val (float): heat capacity

        Returns:
            category (ThermalMassCapacities): category of thermal mass


        """
        if val >= ThermalMassCapacities.Concrete:
            return ThermalMassConstructions.Concrete.value
        elif (
            val < ThermalMassCapacities.Concrete and val >= ThermalMassCapacities.Brick
        ):
            return ThermalMassConstructions.Brick.value
        elif (
            val < ThermalMassCapacities.Brick and val >= ThermalMassCapacities.WoodFrame
        ):
            return ThermalMassConstructions.WoodFrame.value
        elif val < ThermalMassCapacities.WoodFrame:
            return ThermalMassConstructions.SteelFrame.value

    @classmethod
    def single_pane_shgc_estimation(cls, tsol, uval):
        """
        Calculate shgc for single pane window - from Archetypal tsol calulation based on u-val and shgc
        """

        def shgc_intermediate(tsol, uval):
            # if u_factor >= 4.5 and shgc < 0.7206:
            #     return 0.939998 * shgc ** 2 + 0.20332 * shgc
            if uval >= 4.5 and tsol < 0.6346:
                return 10 / 469999 * (math.sqrt(2349995000 * tsol + 25836889) - 5083)
            # if u_factor >= 4.5 and shgc >= 0.7206:
            #     return 1.30415 * shgc - 0.30515
            if uval >= 4.5 and tsol >= 0.6346:
                return (20000 * tsol + 6103) / 26083
            # if u_factor <= 3.4 and shgc <= 0.15:
            #     return 0.41040 * shgc
            if uval <= 3.4 and tsol <= 0.06156:
                return tsol / 0.41040
            # if u_factor <= 3.4 and shgc > 0.15:
            #     return 0.085775 * shgc ** 2 + 0.963954 * shgc - 0.084958
            if uval <= 3.4 and tsol > 0.06156:
                return (
                    -1 * 481977 + math.sqrt(239589100979 + 85775000000 * tsol)
                ) / 85775
            else:
                logger.warning(
                    "Could not calculate shgc - review window parameters. Defaulting to 0.6."
                )
                return 0.6

        if 3.4 <= uval <= 4.5:
            return np.interp(
                uval,
                [3.4, 4.5],
                [shgc_intermediate(tsol, 3.4), shgc_intermediate(tsol, 4.5)],
            )
        else:
            return shgc_intermediate(tsol, uval)

    @classmethod
    def prepare_epw_features(
        cls,
        epw,
        timeseries=[
            "dry_bulb_temperature",
            "dew_point_temperature",
            "relative_humidity",
            "wind_direction",
            "wind_speed",
            "direct_normal_radiation",
            "diffuse_horizontal_radiation",
            "solar_azimuth",
            "solar_elevation",
            "latitude",
            "longitude",
        ],
    ) -> np.ndarray:
        """
        Extracts timeseries data from epw and returns numpy array

        Returns:
            epw_array (np.ndarray): array of timeseries data (n_weather_channels, 8760)
        """
        return extract(epw, timeseries)

    def prepare_shoeboxes(self, shoebox_gen_type):
        """
        Build a dataframe of shoeboxes for the entire scene. This includes the final weighting needed
        to combine the shoeboxes into a single prediction for each building.

        Returns:
            shoeboxes_df (pd.DataFrame): df with shoebox features and weights
        """
        if shoebox_gen_type == "raytrace":
            logger.info("Running raytracer...")
            logger.warning("RAYTRACER NOT SETUP. SKIPPING.")
            shoeboxes_df = self.allocate_shaded_shoeboxes()  # TODO
        elif shoebox_gen_type == "cardinal_unshaded":
            shoeboxes_df = self.allocate_unshaded_cardinal_shoeboxes()
        elif shoebox_gen_type == "edge_unshaded":
            shoeboxes_df = self.allocate_unshaded_edge_shoeboxes()
        elif shoebox_gen_type == "area":
            shoeboxes_df = self.allocate_area_methods_shoeboxess()
        else:
            raise ValueError(
                f"Shoebox generation type {shoebox_gen_type} not supported."
            )

        # # Calculate infiltration TODO: do we want this here? before the merge? Should this be calculated based on shoeboxes or total building?
        # logger.debug(
        #     f"Input infiltration values range between {shoeboxes_df['Infiltration'].max()} and {shoeboxes_df['Infiltration'].min()} ach"
        # )
        # assert (
        #     shoeboxes_df.isna().sum().any() <= 0
        # ), f"{shoeboxes_df.isna().sum().sum()} NaNs found in shoebox df"
        # shoeboxes_df = self.convert_ach_to_infiltration_per_exposed_area(shoeboxes_df)
        errors = shoeboxes_df.isna().sum()
        any_errors = False
        for n, count in errors.items():
            if count > 0:
                logger.error(f"{count} NaNs found in shoebox column {n}.")
                any_errors = True
        if any_errors:
            raise ValueError("NaNs found in shoebox df")

        return shoeboxes_df

    def convert_ach_to_infiltration_per_exposed_area(self, shoeboxes_df):
        """
        Infiltration calculated with:
        I/surface_area = ACH * V / 3.6 / surface_area
        = ACH * (A * h) / 3.6 / (A_facade + A_roof)
        = ACH * (A * h) / 3.6 / (2(w + l) * h + (A_perim_roof + A_core_roof))
        """
        raise ValueError(
            "This needs to be refactored to use the autozoned perimeter areas and core areas rather than shoebox areas"
        )
        logger.info("Recalculating infiltration values for each shoebox.")
        ach = shoeboxes_df["Infiltration"]
        shoeboxes_df["ach"] = ach  # TODO remove this
        w = self.shoebox_width
        h = self.floor_to_floor_height
        p_depth = self.perim_offset  # TODO: or floor_2_facade * height
        c_depths = p_depth * shoeboxes_df["core_2_perim"]
        l = c_depths + p_depth
        a = l * w
        p_roof_areas = shoeboxes_df["roof_2_footprint"] * p_depth * w
        c_roof_areas = shoeboxes_df["roof_2_footprint"] * c_depths * w
        surface_area = 2 * (w + l) * h + p_roof_areas + c_roof_areas

        infiltration_per_exposed_area = ach * a * h / 3600 / surface_area
        # Check for nans
        if infiltration_per_exposed_area.isna().sum() > 0:
            logger.warning("NaNs in infiltration. Replacing with zero.")
            infiltration_per_exposed_area = infiltration_per_exposed_area.fillna(0)
        shoeboxes_df["Infiltration"] = infiltration_per_exposed_area.astype("float64")

        return shoeboxes_df

    def prepare_shading(self):
        pass

    def combine_weights(
        self,
        shoeboxes_df: pd.DataFrame,
        input_keys: List[str],
        output_key: str = "weight",
    ):
        shoeboxes_df[output_key] = shoeboxes_df[input_keys].prod(axis=1)
        return shoeboxes_df

    def pick_weights(
        self,
        shoeboxes_df: pd.DataFrame,
        selector_key: str = "floor_name",
        output_key: str = "floor_weight",
        weight_suffix: str = "_weight",
    ):
        """
        Downselects a weight for each shoebox from the parent building's config based on the selector_key and the weight_suffix
        Assumes that the shoeboxes_df has a list of weights from a parent building, e.g. ["bottom_weight", "middle_weight"] etc
        Which align with a categorical column in the shoeboxes_df, e.g. "floor_name":["bottom", "middle", "middle"] etc

        TODO: this could be more elegant with a melted categorical weight table
        along with a df join


        Args:
            shoeboxes_df (pd.DataFrame): df with shoebox features and weights
            selector_key (str): column name in shoeboxes_df that contains the categorical selector
            output_key (str): column name in shoeboxes_df that will contain the downselected weight
            weight_suffix (str): suffix to append to the selector_key to get the weight column name, e.g. "bottom_weight"

        Returns:
            shoeboxes_df (pd.DataFrame): df with shoebox features and weights

        """
        # Initialize the computed weight to zero
        shoeboxes_df[output_key] = 0.0

        # Get the possible weight types
        selector_values = shoeboxes_df[selector_key].unique()
        # TODO: this could be done with masking rather than a for-loop sum,
        # though the for loop is only over the number of weight categories, e.g. south/east/north/west
        # so it's not too bad
        for selector_value in selector_values:
            # get a vector indicating whether the shoebox is in the correct category
            is_selection = shoeboxes_df[selector_key] == selector_value
            # get the name of the weight for this category
            selector_weight_name = selector_value + weight_suffix
            # multiply the weight by the is_selection mask to zero out the weight if it doesn't match
            # e.g. the shoebox parent building's middle_weight might be 0.33 but if the shoebox is on the bottom floor
            # the selector_weight will return 0.
            selector_weight = shoeboxes_df[selector_weight_name] * is_selection.astype(
                float
            )
            # update the computed weight vector
            shoeboxes_df[output_key] = shoeboxes_df[output_key] + selector_weight
        return shoeboxes_df

    def dimension_shoeboxes(
        self,
        shoeboxes_df: pd.DataFrame,
        perim_to_facade_ratio_method: Literal["building", "theoretical"],
        shoebox_width: float = 3.0,
    ) -> pd.DataFrame:
        """
        Calculates the shoebox dimensions based on the building geometry and the shoebox width
        dynamic shoebox widths can be generated computed based off of edge lengths to make better
        aspect ratio approximations by setting shoebox_width to -1

        Args:
            shoeboxes_df (pd.DataFrame): df with shoebox features and weights
            perim_to_facade_ratio_source (Literal["building", "theoretical"]): whether to use the building's perim/facade ratio or a theoretical autozoner ratio of self.floor_to_floor_height / self.perim_offset
            shoebox_width (float): width of the shoebox, if -1 then the shoebox width will be dynamically comptued from the edge length

        Returns:
            shoeboxes_df (pd.DataFrame): df with shoebox features and weights

        """

        if shoebox_width == -1:
            assert "edge_length" in shoeboxes_df.columns
        else:
            assert shoebox_width > 2, "Shoebox width must be greater than 2 meters"

        if perim_to_facade_ratio_method == "building":
            assert (
                "bldg_perim_area_to_facade_area" in shoeboxes_df.columns
            ), "bldg_perim_area_to_facade_area must be in shoeboxes_df"
        elif perim_to_facade_ratio_method == "theoretical":
            pass
        else:
            raise ValueError(
                f"perim_to_facade_ratio_source must be one of ['building', 'edge']"
            )

        assert type(self.floor_to_floor_height) in [
            float,
            int,
        ], f"floor_to_floor_height must be float; {type(self.floor_to_floor_height)} support not yet implemented"
        assert type(self.perim_offset) in [
            float,
            int,
        ], f"perim_offset must be float; {type(self.perim_offset)} support not yet implemented"
        assert type(shoebox_width) in [
            float,
            int,
        ], f"shoebox_width must be float; {type(shoebox_width)} support not yet implemented"

        shoeboxes_df["height"] = self.floor_to_floor_height

        if perim_to_facade_ratio_method == "building":
            shoeboxes_df[
                "perim_area_to_facade_area"
            ] = shoeboxes_df.bldg_perim_area_to_facade_area

            shoeboxes_df = shoeboxes_df.drop(columns=["bldg_perim_area_to_facade_area"])
        elif perim_to_facade_ratio_method == "theoretical":
            shoeboxes_df["perim_area_to_facade_area"] = (
                self.perim_offset / shoeboxes_df.height
            )

        if shoebox_width > 0:
            shoeboxes_df["width"] = shoebox_width
        else:
            shoeboxes_df["width"] = self.sigmoid_clipping(
                shoeboxes_df,
                column_name="edge_length",
                min_value=2,
                max_value=7,
                steepness=0.1,
                midpoint=30,
            )
            # shoebox width is calculated using a non-linear map based off of edge length
            # in order to get better aspect ratios

        # pop out features for cleaner code
        perim_area_to_facade_area = shoeboxes_df.perim_area_to_facade_area
        bldg_core_area_to_perim_area = shoeboxes_df.bldg_core_area_to_perim_area
        width = shoeboxes_df.width
        height = shoeboxes_df.height

        # compute dimensions
        facade_area = width * height
        perim_area = facade_area * perim_area_to_facade_area
        perim_depth = perim_area / width
        core_depth = perim_depth * bldg_core_area_to_perim_area
        core_area = core_depth * width

        # store dimensions
        shoeboxes_df["facade_area"] = facade_area
        shoeboxes_df["perim_area"] = perim_area
        shoeboxes_df["perim_depth"] = perim_depth
        shoeboxes_df["core_depth"] = core_depth
        shoeboxes_df["core_area"] = core_area
        return shoeboxes_df

    def sigmoid(self, x):
        """
        Implement the sigmoid function using NumPy.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_clipping(
        self, df, column_name, min_value=2, max_value=7, steepness=0.1, midpoint=30
    ):
        """
        Scales the values in a specified column of a DataFrame using a sigmoid function
        such that the values are clipped between `min_value` and `max_value`.
        The inflection point of the sigmoid is set around `midpoint`.
        """
        # Adjusted sigmoid function for clipping with new range and midpoint
        clipped_values = min_value + (max_value - min_value) * self.sigmoid(
            steepness * (df[column_name] - midpoint)
        )
        return clipped_values

    def allocate_unshaded_edge_shoeboxes(self) -> pd.DataFrame:
        sky = Sky(
            epw=self.epw,
            mfactor=4,
            n_azimuths=SHADING_DIV_SIZE * 2,
            dome_radius=200,
            run_conversion=False,
        )
        tracer = Tracer(
            sky=sky,
            gdf=self.building_gdf,
            height_col=self.height_col,
            id_col=self.id_col,
            archetype_col=self.template_name_col,
            node_width=1,
            sensor_spacing=self.sensor_spacing,
            f2f_height=self.floor_to_floor_height,
        )

        tracer.compute_edge_orientation_weights()
        building_idxs: np.ndarray = tracer.edges.building_id.to_numpy()
        edge_qualified_lengths: np.ndarray = tracer.edges.qualified_length.to_numpy()
        edge_weights: np.ndarray = tracer.edges.weight.to_numpy()
        edge_orientations: np.ndarray = (
            tracer.edges.normal_theta.to_numpy() + 2 * np.pi
        ) % (
            2 * np.pi
        )  # wrap around

        n_edges = len(edge_qualified_lengths)
        n_floor_types = len(floor_names)
        shoeboxes_df = pd.DataFrame(
            {
                "building_idx": building_idxs.repeat(n_floor_types),
                "edge_length": edge_qualified_lengths.repeat(n_floor_types),
                "edge_weight": edge_weights.repeat(n_floor_types),
                "orientation": edge_orientations.repeat(n_floor_types),
                "roof_2_footprint": exposure_ratios_roof * n_edges,
                "ground_2_footprint": exposure_ratios_ground * n_edges,
                "floor_name": floor_names * n_edges,
            }
        )

        # TODO: Make sure merge is performed correctly! Building ids from gis_features_df are not the same as building_ids from tracer.edges.building_id
        shoeboxes_df = shoeboxes_df.merge(
            self.gis_features_df, left_on="building_idx", right_on="building_idx"
        )
        logger.debug(
            f"Built sb df from gis features, {shoeboxes_df.isna().sum().sum()} NaNs"
        )
        shoeboxes_df = self.dimension_shoeboxes(
            shoeboxes_df,
            perim_to_facade_ratio_method="building",
            shoebox_width=self.shoebox_width,
        )
        logger.debug(
            f"Assigned sb dimensions in df from gis features, {shoeboxes_df.isna().sum().sum()} NaNs"
        )

        # bring in the template features like heating setpoint, etc
        shoeboxes_df = shoeboxes_df.merge(
            self.template_features_df,
            left_on="template_name",
            right_on=self.template_name_col,
        )
        logger.debug(f"Added template features, {shoeboxes_df.isna().sum().sum()} NaNs")

        shoeboxes_df = self.pick_weights(
            shoeboxes_df,
            selector_key="floor_name",
            output_key="floor_weight",
        )
        shoeboxes_df = self.combine_weights(
            shoeboxes_df,
            input_keys=["edge_weight", "floor_weight"],
            output_key="weight",
        )

        logger.debug("Dropping unnecessary shoeboxes...")
        shoeboxes_df = shoeboxes_df[(shoeboxes_df.weight - 0).abs() > 1e-6]

        shading_cols = [f"shading_{x}" for x in range(SHADING_DIV_SIZE)]
        shoeboxes_df[shading_cols] = 0.0
        logger.debug(f"Shoeboxes built... {shoeboxes_df.isna().sum().sum()} NaNs")
        return shoeboxes_df

    def allocate_unshaded_cardinal_shoeboxes(self) -> pd.DataFrame:
        """
        Returns:
            shoebox_df (pd.DataFrame): df with sizing details, building_id, orientation, and template_name, template_idx, joined with template_df that has template_dict column names, etc
        """
        sky = Sky(
            epw=self.epw,
            mfactor=4,
            n_azimuths=SHADING_DIV_SIZE * 2,
            dome_radius=200,
            run_conversion=False,
        )
        tracer = Tracer(
            sky=sky,
            gdf=self.building_gdf,
            height_col=self.height_col,
            id_col=self.id_col,
            archetype_col=self.template_name_col,
            node_width=1,
            sensor_spacing=self.sensor_spacing,
            f2f_height=self.floor_to_floor_height,
        )

        # Compute and store the N/E/S/W weights for each building
        tracer.compute_edge_orientation_weights()
        self.gis_features_df["north_weight"] = tracer.buildings.north_weight.to_numpy()
        self.gis_features_df["east_weight"] = tracer.buildings.east_weight.to_numpy()
        self.gis_features_df["south_weight"] = tracer.buildings.south_weight.to_numpy()
        self.gis_features_df["west_weight"] = tracer.buildings.west_weight.to_numpy()

        n_buildings = len(self.gis_features_df)

        # TODO: check orientation # ordinal mapping
        shoeboxes_df = pd.DataFrame(
            {
                "building_id": self.gis_features_df["building_id"].values.repeat(16),
                "orientation": sorted(["east", "north", "south", "west"] * 4)
                * n_buildings,
                "roof_2_footprint": ([0.0, 0.0, 1.0, 1.0] * 4) * n_buildings,
                "ground_2_footprint": ([1.0, 0.0, 0.0, 1.0] * 4) * n_buildings,
                "floor_name": (floor_names * 4) * n_buildings,
            }
        )
        # Bring in the building's geometric features like core_2_perim and floor_2_facade, height, etc
        shoeboxes_df = shoeboxes_df.merge(
            self.gis_features_df, left_on="building_id", right_on="building_id"
        )
        shoeboxes_df = self.dimension_shoeboxes(
            shoeboxes_df,
            perim_to_facade_ratio_method="building",
            shoebox_width=self.shoebox_width,
        )
        logger.debug(
            f"Built sb df from gis features, {shoeboxes_df.isna().sum().sum()} NaNs"
        )
        # bring in the template features like heating setpoint, etc
        shoeboxes_df = shoeboxes_df.merge(
            self.template_features_df,
            left_on="template_name",
            right_on=self.template_name_col,
        )
        logger.debug(f"Added template features, {shoeboxes_df.isna().sum().sum()} NaNs")

        shoeboxes_df = self.pick_weights(
            shoeboxes_df, selector_key="orientation", output_key="oriented_weight"
        )

        shoeboxes_df = self.pick_weights(
            shoeboxes_df, selector_key="floor_name", output_key="floor_weight"
        )

        shoeboxes_df = self.combine_weights(
            shoeboxes_df,
            input_keys=["oriented_weight", "floor_weight"],
            output_key="weight",
        )

        logger.debug("Dropping unnecessary shoeboxes...")
        shoeboxes_df = shoeboxes_df[(shoeboxes_df.weight - 0).abs() > 1e-6]

        shading_cols = [f"shading_{x}" for x in range(SHADING_DIV_SIZE)]
        shoeboxes_df[shading_cols] = 0.0
        logger.debug(f"Shoeboxes built... {shoeboxes_df.isna().sum().sum()} NaNs")
        return shoeboxes_df

    def allocate_area_methods_shoeboxess(self) -> pd.DataFrame:
        """
        A method for calculating building energy use based on a standard south-facing shoebox,
        multiplied out by the total building area.
        """
        shoeboxes_df = pd.DataFrame(
            {
                "building_id": self.gis_features_df["building_id"].values,
                "orientation": "south",
                "roof_2_footprint": 1 / self.gis_features_df["floor_count"],
                "ground_2_footprint": 1 / self.gis_features_df["floor_count"],
            }
        )
        # Bring in the building's geometric features like core_2_perim and floor_2_facade, height, etc
        gis_with_heights = self.gis_features_df
        gis_with_heights["height"] = self.floor_to_floor_height
        shoeboxes_df = shoeboxes_df.merge(
            gis_with_heights, left_on="building_id", right_on="building_id"
        )
        shoeboxes_df = shoeboxes_df.merge(
            self.template_features_df,
            left_on="template_name",
            right_on=self.template_name_col,
        )
        shoeboxes_df["weight"] = 1
        # apply zero shading
        shading_cols = [f"shading_{x}" for x in range(SHADING_DIV_SIZE)]
        shoeboxes_df[shading_cols] = 0.0

        logger.debug(
            f"Built simple area method shoeboxes, with shape {shoeboxes_df.shape}"
        )
        return shoeboxes_df

    def allocate_shaded_shoeboxes(self) -> pd.DataFrame:
        """
        Future
        Returns:
            shoeboxes_df (pd.DataFrame): df with sizing details, building_id, orientation, and template_name, template_idx, joined with template_df that has template_dict column names, etc

        """
        pass

    def visualize_2d(self, ax=None, gdf=None, max_polys=1000, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        if gdf is not None:
            max_idx = min(gdf.shape[0], max_polys)
            gdf.iloc[:max_idx].plot(aspect=1, **kwargs)
        else:
            max_idx = min(self.building_gdf.shape[0], max_polys)
            self.building_gdf.iloc[:max_idx].plot(
                column=self.template_name_col, ax=ax, aspect=1, **kwargs
            )
            try:
                self.building_gdf.iloc[:max_idx]["cores"].plot(
                    facecolor="none", edgecolor="red", ax=ax, aspect=1, **kwargs
                )
            except:
                logger.error("Failed to plot cores!")
                pass
        ax.axis("off")
        return ax

    def prepare_for_surrogate(self):
        features = self.shoeboxes_df.copy(deep=True)
        # Convert orientations to numerical
        return features, self.schedules_array, self.epw_array

    @classmethod
    def open_uio(
        cls,
        filename,
        epw_path=None,
        archetypal_path=None,
        template_map=None,
        results_df=None,
        wwr=0.4,
        shoebox_gen_type="unshaded",
    ):
        filename = Path(filename)
        logger.info("Opening UIO model...")

        # TODO fetch epw based on lat and long centroid?

        # 1. Read the UIO zipfile Needs a temp directory because cannot be read in memory.
        with tempfile.TemporaryDirectory() as tempdir:
            logger.debug(f"TEMPORARY LOCATION {tempdir}")
            with ZipFile(filename, "r") as umizip:
                json_file, *_ = (
                    file for file in umizip.namelist() if file.endswith(".json")
                )
                umizip.extract(json_file, tempdir)
                with open(Path(tempdir, json_file), "r") as f:
                    keyfields = json.load(f)
                giszip_filename, *_ = (
                    file for file in umizip.namelist() if file.endswith(".zip")
                )
                start_time = time.time()
                logger.info("reading input file...")
                umizip.extract(giszip_filename, tempdir)
                gdf = gpd.read_file(Path(tempdir, giszip_filename))
                logger.info(
                    f"Read {gdf.memory_usage(index=True).sum() / 1000:,.1f}KB from"
                    f" {filename} in"
                    f" {time.time() - start_time:,.2f} seconds"
                )

                # 2. Parse the weather file as :class:`Epw`
                try:
                    epw = EPW(epw_path)
                    logger.info(f"Successfully loaded {epw}")
                except:
                    raise ImportError(f"Error opening EPW {epw_path}.")

                # 3. Parse the templates library.
                try:
                    logger.info(f"Opening archetpal templates at {archetypal_path}")
                    with open(Path(archetypal_path), "r") as f:
                        template_lib_json = json.load(f)
                        template_lib = init_template(template_lib_json)
                except:
                    raise ImportError(
                        f"Error opening Archetypal template at {archetypal_path}."
                    )

                # 4. Check the templates against the gdf and remap gdf template names
                # TODO: mapping of templates to gdf
                # Add new column for template ID based on primary and secondary divisions
                # if keyfields["primary"] != "N/A":
                #     if keyfields["secondary"] != "N/A":
                #         gdf["template_name"] = gdf[
                #             [keyfields["primary"], keyfields["secondary"]]
                #         ].agg("_".join, axis=1)
                #     else:
                #         gdf["template_name"] = gdf[keyfields["primary"]]
                dummy_template_name_col = "template_name"
                dummy_wwr_col_name = "wwr"
                gdf[dummy_template_name_col] = np.random.choices(
                    [t.Name for t in template_lib.BuildingTemplates], size=gdf.shape[0]
                )
                umi_gdf = gdf[
                    [
                        "geometry",
                        keyfields["height"],
                        keyfields["id"],
                        dummy_template_name_col,
                    ]
                ]
                umi_gdf[dummy_wwr_col_name] = wwr

            return cls(
                gdf=umi_gdf,
                id_col=keyfields["id"],
                height_col=keyfields["height"],
                wwr_col=dummy_wwr_col_name,
                template_name_col=dummy_template_name_col,
                template_lib=template_lib,
                epw=epw,
                shoebox_gen_type=shoebox_gen_type,
            )

    @classmethod
    def open_umi(
        cls,
        filename,
        height_col="HEIGHT",
        id_col="guid",
        template_name_col="TemplateName",
        wwr_col="WindowToWallRatioN",
        shoebox_gen_type="edge_unshaded",
    ):
        """
        WARNING: THIS CURRENTLY ONLY WORKS WITH SIMULATED UMI FILES (shoebox weights and gdf are only created in sdl_common post simulation)
        """
        # 1. Read the UMI zipfile Needs a temp directory because cannot be read in memory.
        logger.info("reading input file...")
        with tempfile.TemporaryDirectory() as tempdir:
            logger.debug(f"TEMPORARY LOCATION {tempdir}")
            with ZipFile(filename, "r") as umizip:
                # 2. Parse the weather file as :class:`Epw`
                epw_file, *_ = (
                    file for file in umizip.namelist() if file.endswith(".epw")
                )
                p = umizip.extract(epw_file, tempdir)
                try:
                    epw = EPW(p)
                    logger.info(f"Successfully loaded {epw}")
                except:
                    raise ImportError(f"Error opening EPW {epw_file}.")

                # 3. Parse the templates library.
                logger.debug(umizip.namelist())
                try:
                    template_file, *_ = (
                        file
                        for file in umizip.namelist()
                        if file.endswith(".json") and "sdl-common" not in file
                    )
                    logger.info(f"Opening archetpal templates at {template_file}")
                    umizip.extract(template_file, tempdir)
                    with open(Path(tempdir, template_file), "r") as f:
                        template_lib_json = json.load(f)
                        template_lib = init_template(template_lib_json)
                except:
                    raise ImportError(
                        f"Error opening Archetypal template at {template_file}."
                    )

                # 5. Parse all the .json files in "sdl-common" folder
                sdl_common = {}  # prepare sdl_common dict
                # loop over 'sdl-common' config files (.json)
                for file in [
                    file for file in umizip.infolist() if "sdl-common" in file.filename
                ]:
                    if file.filename.endswith("project.json"):
                        start_time = time.time()
                        # This is the geojson representation of the project.
                        # First, figure out the utm_crs for the weather location
                        lat, lon = epw.location.latitude, epw.location.longitude
                        utm_zone = int(math.floor((float(lon) + 180) / 6.0) + 1)
                        utm_crs = CRS.from_string(
                            f"+proj=utm +zone={utm_zone} +ellps=WGS84 "
                            f"+datum=WGS84 +units=m +no_defs"
                        )
                        # Second, load the GeoDataFrame TODO: check CRS for area calcs
                        with umizip.open(file) as gdf_f:
                            gdf = GeoDataFrame.from_file(gdf_f)
                            gdf._crs = utm_crs
                            logger.debug(gdf.columns)
                        logger.info(
                            f"Read {gdf.memory_usage(index=True).sum() / 1000:,.1f}KB from"
                            f" {filename} in"
                            f" {time.time() - start_time:,.2f} seconds"
                        )
                    # elif file.filename.endswith("energy.zip"):
                    # # We load the IDF models
                    # with umizip.open(file) as zfiledata:
                    #     with ZipFile(zfiledata) as energy_zip:
                    #         for sample in energy_zip.infolist():
                    #             shoeboxes[
                    #                 str(Path(sample.filename).expand())
                    #             ] = ShoeBox(
                    #                 StringIO(energy_zip.open(sample).read().decode()),
                    #                 epw=StringIO(epw.to_file_string()),
                    #                 as_version="8.4",
                    #             )
                    else:
                        with umizip.open(file) as f:
                            try:
                                sdl_common[
                                    Path(file.filename.replace("\\", "/")).stem
                                ] = json.load(f)
                            except JSONDecodeError:  # todo: deal with xml
                                sdl_common[
                                    Path(file.filename.replace("\\", "/")).stem
                                ] = {}

                # 4. Check the templates against the gdf and remap gdf template names
                # TODO: mapping of templates to gdf
                try:
                    f2f_height = gdf["FloorToFloorHeight"][0]
                except:
                    logger.info("Calculating individual floor-to-floor heights.")
                    logger.debug(f"Min building height: {gdf[height_col].min()}")
                    logger.debug(f"Max floor count: {gdf['FloorCount'].min()}")
                    f2f_height = gdf[height_col] / gdf["FloorCount"]
                    gdf["FloorToFloorHeight"] = f2f_height
                    logger.debug(
                        f"Floor-to-floor height: {f2f_height.min()} to {f2f_height.max()}"
                    )
                    logger.warning(
                        "Dynamic floor to floor height found, but resetting to 3."
                    )
                    f2f_height = 3
                try:
                    perim_offset = gdf["PerimeterOffset"][0]
                except:
                    logger.warning(
                        f"No perimeter offset in gdf, defaulting to {PERIM_OFFSET}"
                    )
                    perim_offset = PERIM_OFFSET
                try:
                    width = gdf["RoomWidth"][0]
                except:
                    logger.warning(
                        f"No shoebox width in gdf, defaulting to 3 (dynamic width))"
                    )
                    width = 3
                umi_gdf = gdf[
                    ["geometry", height_col, id_col, template_name_col, wwr_col]
                ]

                errors = umi_gdf.isna().sum()
                for n, count in errors.items():
                    if count > 0:
                        logger.warning(f"{count} NaNs found in GDF column {n}.")
                        raise ValueError

            return cls(
                gdf=umi_gdf,
                height_col=height_col,
                wwr_col=wwr_col,
                id_col=id_col,
                template_name_col=template_name_col,
                template_lib=template_lib,
                epw=epw,
                shoebox_width=width,
                floor_to_floor_height=f2f_height,
                perim_offset=perim_offset,
                shoebox_gen_type=shoebox_gen_type,
            )


if __name__ == "__main__":
    # umi_test = Umi.open_uio(
    #     filename="D:/Users/zoelh/GitRepos/ml-for-building-energy-modeling/umi_data/Florianopolis/Florianopolis_Baseline.uio",
    #     epw_path="ml-for-bem/data/epws/global_epws_indexed/cityidx_0033_BRA_SP-São Paulo-Congonhas AP.837800_TRY Brazil.epw",
    #     archetypal_path="ml-for-bem/data/template_libs/cz_libs/residential/CZ3A.json",
    # )
    gdf = gpd.read_file(Path("data") / "gis" / "Florianopolis_Baseline.zip")

    # dict to store key fields for known gis files
    id_cols = {
        "florianpolis": {
            "height_col": "HEIGHT",
            "id_col": "OBJECTID",
            "template_name_col": "template_name",
            "wwr_col": "wwr",
        }
    }

    epw_fp = (
        Path("data")
        / "epws"
        / "global_epws_indexed"
        / "cityidx_0033_BRA_SP-São Paulo-Congonhas AP.837800_TRY Brazil.epw"
    )

    epw = EPW(epw_fp)
    template_lib = UmiTemplateLibrary.open(
        Path("data") / "template_libs" / "BostonTemplateLibrary.json"
    )

    # Insert dummy template names
    gdf[id_cols["florianpolis"]["template_name_col"]] = np.random.choice(
        [t.Name for t in template_lib.BuildingTemplates], size=gdf.shape[0]
    )
    # insert dummy wwrs
    gdf[id_cols["florianpolis"]["wwr_col"]] = 0.4

    umi_test = UBEM(
        gdf=gdf,
        **id_cols["florianpolis"],
        epw=epw,
        template_lib=template_lib,
        shoebox_width=3,
        floor_to_floor_height=4,
        perim_offset=4,
    )

    print("SCHEDULES ARRAY: ", umi_test.schedules_array.shape)
    print("TEMPLATE DF: ", umi_test.template_features_df.shape)
    print("EPW ARRAY: ", umi_test.epw_array.shape)
    print(umi_test.shoeboxes_df[["core_2_perim", "floor_2_facade"]])

    umi_test = UBEM.open_umi("data/Braga_Baseline.umi", height_col="height (m)")
    print(umi_test.shoeboxes_df)

# TODO:
# Shading vector / RayTracing
# mapping
# using numerics for orientation instead of words?
# infiltration/ventilation etc
# enable providing a template_df directly in addition to a UTL object
