from typing import Tuple
import sys
import os
from pathlib import Path
from zipfile import ZipFile, ZipInfo
import json
import tempfile
import logging
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from ladybug.epw import EPW
import geopandas as gpd
from geopandas import GeoDataFrame
from pyproj import CRS
from json import JSONDecodeError

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
from archetypal.template.schedule import (
    DaySchedule,
    WeekSchedule,
    YearSchedule,
)
from archetypal.template.structure import StructureInformation
from archetypal.template.ventilation import VentilationSetting
from archetypal.template.window_setting import WindowSetting
from archetypal.template.zone_construction_set import ZoneConstructionSet
from archetypal.template.zonedefinition import ZoneDefinition

from shoeboxer.builder import template_dict
from utils.constants import *
from utils.schedules import get_schedules
from weather.weather import extract
from umi.trace import Tracer, Sky

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


class Umi:
    def __init__(
        self,
        gdf: GeoDataFrame,
        height_col: str,
        wwr_col: str,
        id_col: str,
        template_name_col: str,
        epw: EPW,
        template_lib: UmiTemplateLibrary,
        shoebox_width=3,  # Can be list (for each bldg) or single value TODO
        floor_to_floor_height=4,  # Can be list (for each bldg) or single value TODO,
        perim_offset=PERIM_OFFSET,
        calculate_shading=False,
    ):
        assert len([t.Name for t in template_lib.BuildingTemplates]) == len(
            set([t.Name for t in template_lib.BuildingTemplates])
        ), "Duplicate names in template library!  Aborting."

        # Store column metadata for building_gdf
        self.height_col = height_col
        self.id_col = id_col
        self.wwr_col = wwr_col
        self.template_name_col = template_name_col
        self.template_idx_col = "template_idx"

        # store objects
        self.building_gdf = gdf
        self.epw = epw
        self.template_lib = template_lib

        # set geometric variables
        self.shoebox_width = shoebox_width
        self.floor_to_floor_height = floor_to_floor_height
        self.perim_offset = perim_offset

        start_time = time.time()
        (
            self.template_features_df,
            self.schedules_array,
        ) = self.prepare_archetypal_features(self.template_lib)
        # Add in template_idx for feature array
        self.building_gdf[self.template_idx_col] = list(
            self.template_features_df.loc[self.building_gdf[self.template_name_col]][
                self.template_idx_col
            ]
        )
        self.epw_array = self.prepare_epw_features()
        self.gis_features_df = self.prepare_gis_features()
        self.shoeboxes_df = self.prepare_shoeboxes(calculate_shading)

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

        perim_areas = footprint_areas - core_areas
        core_2_perim = core_areas / perim_areas
        facade_area = perimeter_length * heights
        floor_2_facade = perim_areas / facade_area
        floor_count = round(heights / self.floor_to_floor_height)
        # TODO: reset floor_to_floor height so there is a whole number of floors?

        geometric_features_df = pd.DataFrame()
        geometric_features_df["building_id"] = ids
        geometric_features_df["wwr"] = wwrs
        geometric_features_df["height"] = heights
        geometric_features_df["footprint_area"] = footprint_areas
        geometric_features_df["core_2_perim"] = core_2_perim
        geometric_features_df["floor_2_facade"] = floor_2_facade
        geometric_features_df["floor_count"] = floor_count
        geometric_features_df["template_name"] = template_names

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

        n_templates = len(template_lib.BuildingTemplates)
        schedules = np.zeros((n_templates, len(SCHEDULE_PATHS), 8760))
        for i, (_, d) in enumerate(template_data_dict.items()):
            schedules[i] = d.pop("schedules")
        # TODO: many of these are not returning with correct dtypes
        template_features = pd.DataFrame.from_dict(template_data_dict).T
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
            vent_mode = MechVentMode.AllOn
        elif "ALWAYSON" in vent_sched_name.upper():
            vent_mode = MechVentMode.OccupancySchedule
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
            return ThermalMassConstructions.Concrete
        elif (
            val < ThermalMassCapacities.Concrete and val >= ThermalMassCapacities.Brick
        ):
            return ThermalMassConstructions.Brick
        elif (
            val < ThermalMassCapacities.Brick and val >= ThermalMassCapacities.WoodFrame
        ):
            return ThermalMassConstructions.WoodFrame
        elif val < ThermalMassCapacities.WoodFrame:
            return ThermalMassConstructions.SteelFrame

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

    def prepare_epw_features(
        self,
        timeseries=[
            "dry_bulb_temperature",
            "dew_point_temperature",
            "relative_humidity",
            "wind_direction",
            "wind_speed",
            "direct_normal_radiation",
            "diffuse_horizontal_radiation",
        ],
    ) -> np.ndarray:
        """
        Extracts timeseries data from epw and returns numpy array

        Returns:
            epw_array (np.ndarray): array of timeseries data (n_weather_channels, 8760)
        """
        return extract(self.epw, timeseries)

    def prepare_shoeboxes(self, calculate_shading):
        """
        Build a dataframe of shoeboxes for the entire scene. This includes the final weighting needed
        to combine the shoeboxes into a single prediction for each building.

        Returns:
            shoeboxes_df (pd.DataFrame): df with shoebox features and weights
        """
        if calculate_shading:
            logger.info("Running raytracer...")
            logger.warning("RAYTRACER NOT SETUP. SKIPPING.")
            shoeboxes_df = self.allocate_shaded_shoeboxes()  # TODO
        else:
            # Make
            shoeboxes_df = self.allocate_unshaded_shoeboxes()
        return shoeboxes_df

    def prepare_shading(self):
        pass

    def calculate_weights(self):
        """
        Append to buildings_dataframe shoebox weight info based on polygon edge proportions and normals ["N_weight", "E_weight", "S_weight", "W_weight"]
        """
        # TODO: Using dummy weights for future integration with Tracer utils
        # Calculate lengths and normals of all polygon edges
        # Cluster to nearest cardinal direction
        weights_df = pd.DataFrame()
        weights_df["guid"] = self.building_gdf["guid"]
        weights_df["North"] = 0.1
        weights_df["East"] = 0.2
        weights_df["West"] = 0.2
        weights_df["South"] = 0.5
        return weights_df

    def allocate_unshaded_shoeboxes(self) -> pd.DataFrame:
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
            sensor_spacing=self.shoebox_width,
            f2f_height=self.floor_to_floor_height,
        )

        # Compute and store the N/E/S/W weights for each building
        tracer.compute_edge_orientation_weights()
        self.gis_features_df["north_weight"] = tracer.buildings.north_weight.to_numpy()
        self.gis_features_df["east_weight"] = tracer.buildings.east_weight.to_numpy()
        self.gis_features_df["south_weight"] = tracer.buildings.south_weight.to_numpy()
        self.gis_features_df["west_weight"] = tracer.buildings.west_weight.to_numpy()

        # Compute the floor weight for each building
        # If there is more than 1 floor, then we use the four shoeboxes from the bottom floor exactly once
        # and the four shoeboxes from the top floor exactly once
        # If there are more than 2 floors, then we use the four shoeboxes from the middle floors floor_count - 2 times
        # Otherwise if there is exactly 1 floor, then we use the Exclusive shoeboxes for each direction exactly once
        floor_names = ["bottom", "middle", "top", "exclusive"]
        self.gis_features_df["bottom"] = (
            self.gis_features_df["floor_count"] > 1
        ).astype(int)
        self.gis_features_df["top"] = (self.gis_features_df["floor_count"] > 1).astype(
            int
        )
        self.gis_features_df["middle"] = (
            self.gis_features_df["floor_count"] > 2
        ).astype(int) * (self.gis_features_df["floor_count"] - 2)

        self.gis_features_df["exclusive"] = (
            self.gis_features_df["floor_count"] == 1
        ).astype(int)
        floor_counts = self.gis_features_df["floor_count"].values.reshape(-1, 1)

        # normalize floor_totals
        self.gis_features_df[floor_names] = (
            self.gis_features_df[floor_names].values / floor_counts
        ).clip(0, 1)

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
        # bring in the template features like heating setpoint, etc
        shoeboxes_df = shoeboxes_df.merge(
            self.template_features_df,
            left_on="template_name",
            right_on=self.template_name_col,
        )

        # calculate the shoebox weights
        # the shoebox weights are the product of the orientation weight and the floor weight
        shoeboxes_df["oriented_weight"] = 0
        shoeboxes_df["floor_weight"] = 0

        # select the weight for each shoeboxes correct orientation
        # Iterate over the four orientations
        for orientation in shoeboxes_df.orientation.unique():
            # make a mask for all the shoeboxes that match this orientation
            is_aligned = shoeboxes_df["orientation"] == orientation
            weight_key = orientation.lower() + "_weight"
            # multiply the orientation weight by the is_aligned mask to zero out the weight
            # if it is not aligned with an orientation
            orientation_weight = shoeboxes_df[weight_key] * is_aligned.astype(float)
            # update the oriented_weight vector
            shoeboxes_df["oriented_weight"] = (
                shoeboxes_df["oriented_weight"] + orientation_weight
            )

        # do the same thing for the floors
        for floor_name in floor_names:
            is_on_floor = shoeboxes_df["floor_name"] == floor_name
            floor_weight = shoeboxes_df[floor_name] * is_on_floor.astype(float)
            shoeboxes_df["floor_weight"] = shoeboxes_df["floor_weight"] + floor_weight

        # multiply the oriented_weight and floor_weight to get the final weight
        # final weights for a building sum to 1
        # the oriented weight determines how much a shoebox contributes to the prediction for a single floor
        # of a building
        # and the floor_weight determines how much all the shoeboxes on that floor contribute to the whole building
        shoeboxes_df["weight"] = (
            shoeboxes_df["oriented_weight"] * shoeboxes_df["floor_weight"]
        )

        logger.debug("Dropping unnecessary shoeboxes...")
        shoeboxes_df = shoeboxes_df[(shoeboxes_df.weight - 0).abs() > 1e-6]

        shading_cols = [f"shading_{x}" for x in range(SHADING_DIV_SIZE)]
        shoeboxes_df[shading_cols] = 0.0
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
        if gdf is None:
            max_idx = min(gdf.shape[0], max_polys)
            gdf.iloc[:max_idx].plot(**kwargs)
        else:
            max_idx = min(self.building_gdf.shape[0], max_polys)
            self.building_gdf.iloc[:max_idx].plot(
                column="template_name", **kwargs, ax=ax
            )
            try:
                self.building_gdf.iloc[:max_idx]["cores"].plot(
                    facecolor="none", edgecolor="red", ax=ax
                )
            except:
                logger.error("Failed to plot cores!")
                pass
        ax.axis("off")
        return ax

    @classmethod
    def open_uio(
        cls,
        filename,
        epw_path=None,
        archetypal_path=None,
        template_map=None,
        results_df=None,
        wwr=0.4,
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
                gdf[dummy_template_name_col] = np.random.randint(
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
            )

    @classmethod
    def open_umi(cls, filename):
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
                try:
                    template_file, *_ = (
                        file
                        for file in umizip.namelist()
                        if file.endswith("template-library.json")
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
                f2f_height = gdf["FloorToFloorHeight"][0]
                perim_offset = gdf["PerimeterOffset"][0]
                width = gdf["RoomWidth"][0]
                height_col = "HEIGHT"
                id_col = "guid"
                template_name_col = "TemplateName"
                wwr_col = "WindowToWallRatioN"
                umi_gdf = umi_gdf[
                    ["geometry", height_col, id_col, template_name_col, wwr_col]
                ]

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

    umi_test = Umi(
        gdf=gdf,
        **id_cols["florianpolis"],
        epw=epw,
        template_lib=template_lib,
        shoebox_width=3,
        floor_to_floor_height=4,
        perim_offset=PERIM_OFFSET,
    )

    print("SCHEDULES ARRAY: ", umi_test.schedules_array.shape)
    print("TEMPLATE DF: ", umi_test.template_features_df.shape)
    print("EPW ARRAY: ", umi_test.epw_array.shape)

# TODO:
# Shading vector / RayTracing
# mapping
# using numerics for orientation instead of words?
# infiltration/ventilation etc
# enable providing a template_df directly in addition to a UTL object
