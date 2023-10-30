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

# module_path = os.path.abspath(
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# )
# if module_path not in sys.path:
#     sys.path.append(str(module_path))

from shoeboxer.builder import template_dict
from utils.constants import *
from schedules import get_schedules
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


class Umi:
    def __init__(
        self,
        gdf: GeoDataFrame,
        epw: EPW,
        template_lib: UmiTemplateLibrary,
        shoebox_width=3,  # Can be list (for each bldg) or single value TODO
        floor_to_floor_height=4,  # Can be list (for each bldg) or single value TODO,
        perim_offset=PERIM_OFFSET,
    ):
        self.gdf = gdf
        self.epw = epw
        self.template_lib = template_lib

        # set geometric variables
        self.shoebox_width = shoebox_width
        self.floor_to_floor_height = floor_to_floor_height
        self.perim_offset = perim_offset

        start_time = time.time()
        self.schedules_array, self.features_df = self.prepare_archetype_features()
        self.epw_array = self.prepare_epw_features()
        self.prepare_gis_features()
        logger.info(f"Processed UMI in {time.time() - start_time:,.2f} seconds")

    def prepare_gis_features(self):
        self.gdf["footprint_area"] = self.gdf["geometry"].area
        self.gdf["cores"] = self.gdf["geometry"].buffer(-1 * self.perim_offset)
        core_areas = self.gdf["cores"].area
        perim_areas = self.gdf["footprint_area"] - core_areas
        self.gdf["core_2_perim"] = core_areas / perim_areas
        perimeter_length = self.gdf["geometry"].length
        facade_area = perimeter_length * self.gdf["height"]
        self.gdf["floor_2_facade"] = perim_areas / facade_area
        self.gdf["floor_count"] = round(
            self.gdf["height"] / self.floor_to_floor_height
        )  # TODO reset floor_to_floor height so there is a whole number of floors?
        self.gdf["roof_2_footprint"] = (
            1 / self.gdf["floor_count"]
        )  # fooprint_A / TFA = fooprint_A/(fooprint_A * n_floors)
        self.gdf["ground_2_footprint"] = self.gdf[
            "roof_2_footprint"
        ]  # Always the same for 2.5D

    def prepare_archetype_features(self):
        """
        Fetches data from archetypal building templates for shoeboxes.

        Returns:
            schedules:  a numpy array of schedule data for each building template [n_templates, n_used_templates (3), 8760]
            template_df: a pandas df of template features that are used in the template_dict of the shoebox builder (and surrogate)
        """
        # TODO pass over unused templates
        template_vectors_dict = {}
        for building_template in self.template_lib.BuildingTemplates:
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
                    wall_mass = self.sort_tmass(constr.heat_capacity_per_unit_wall_area)
                    logger.debug(
                        f"Found facade with r_value {round(wall_r, 2)} and tmass bin {wall_mass}"
                    )
                if name == "Roof":
                    roof_r = constr.r_value
                    roof_mass = self.sort_tmass(constr.heat_capacity_per_unit_wall_area)
                    logger.debug(
                        f"Found roof with r_value {round(roof_r, 2)} and tmass bin {roof_mass}"
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
                shgc = self.single_pane_shgc_estimation(tsol, window_u)

            if shgc > 1:
                logger.warning("SHGC over 1, clipping.")
                shgc = 1.0

            # 4. Get the ventilation factors
            # TODO: how to deal with things that are named weird
            vent_sched_name = (
                building_template.Perimeter.Conditioning.MechVentSchedule.Name
            )
            if building_template.Perimeter.Conditioning.MechVentSchedule == 0:
                vent_mode = MechVentMode.Off.value
            elif "ALWAYSOFF" in vent_sched_name.upper():
                vent_mode = MechVentMode.AllOn.value
            elif "ALWAYSON" in vent_sched_name.upper():
                vent_mode = MechVentMode.OccupancySchedule.value
            elif "OCC" in vent_sched_name.upper():
                vent_mode = MechVentMode.OccupancySchedule.value
            else:
                logger.warning(
                    "Mechanical ventilation response for schedule {vent_sched_name} is not supported. Defaulting to occupancy."
                )
                vent_mode = MechVentMode.OccupancySchedule.value

            recovery_type = (
                building_template.Perimeter.Conditioning.HeatRecoveryType.name
            )
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
                people_density=building_template.Perimeter.Loads.PeopleDensity,
                lighting_power_density=building_template.Perimeter.Loads.LightingPowerDensity,
                equipment_power_density=building_template.Perimeter.Loads.EquipmentPowerDensity,
                infiltration_per_area=building_template.Perimeter.Ventilation.Infiltration,
                ventilation_per_floor_area=building_template.Perimeter.Conditioning.MinFreshAirPerArea,
                ventilation_per_person=building_template.Perimeter.Conditioning.MinFreshAirPerPerson,
                ventilation_mode=vent_mode,
                heating_sp=building_template.Perimeter.Conditioning.HeatingSetpoint,
                cooling_sp=building_template.Perimeter.Conditioning.CoolingSetpoint,
                heat_recovery=getattr(HRV, recovery_type).value,
                economizer=getattr(Econ, econ_type).value,
                wall_r_val=wall_r,
                wall_mass=wall_mass,
                roof_r_val=roof_r,
                roof_mass=roof_mass,
                slab_r_val=slab_r,
                shgc=shgc,
                window_u_val=window_u,
            )
            template_vectors_dict[building_template.Name] = td

        n_templates = len(self.template_lib.BuildingTemplates)
        schedules = np.zeros((n_templates, len(SCHEDULE_PATHS), 8760))
        for i, (_, d) in enumerate(template_vectors_dict.items()):
            schedules[i] = d.pop("schedules")
        return schedules, pd.DataFrame.from_dict(template_vectors_dict).T

    def sort_tmass(self, val):
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

    def single_pane_shgc_estimation(self, tsol, uval):
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
    ):
        return extract(self.epw, timeseries)

    def visualize_2d(self, ax=None, gdf=None, max_polys=1000, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        if gdf:
            max_idx = min(gdf.shape[0], max_polys)
            gdf.iloc[:max_idx].plot(**kwargs)
        else:
            max_idx = min(self.gdf.shape[0], max_polys)
            self.gdf.iloc[:max_idx].plot(column="template", **kwargs, ax=ax)
            try:
                self.gdf.iloc[:max_idx]["cores"].plot(
                    facecolor="none", edgecolor="red", ax=ax
                )
            except:
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
                #         gdf["template"] = gdf[
                #             [keyfields["primary"], keyfields["secondary"]]
                #         ].agg("_".join, axis=1)
                #     else:
                #         gdf["template"] = gdf[keyfields["primary"]]
                gdf["template"] = np.random.randint(
                    len(template_lib.BuildingTemplates), size=gdf.shape[0]
                )
                umi_gdf = gdf[[keyfields["height"], "geometry", "template"]]
                umi_gdf.columns = ["height", "geometry", "template"]

            return cls(
                gdf=umi_gdf,
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
                # Add new column for template ID based on primary and secondary divisions
                # if keyfields["primary"] != "N/A":
                #     if keyfields["secondary"] != "N/A":
                #         gdf["template"] = gdf[
                #             [keyfields["primary"], keyfields["secondary"]]
                #         ].agg("_".join, axis=1)
                #     else:
                #         gdf["template"] = gdf[keyfields["primary"]]
                f2f_height = gdf["FloorToFloorHeight"][0]
                perim_offset = gdf["PerimeterOffset"][0]
                width = gdf["RoomWidth"][0]
                umi_gdf = gdf[
                    [
                        "HEIGHT",
                        "geometry",
                        "TemplateName",
                        "WindowToWallRatioN",
                    ]
                ]
                # "ArchetypeID"
                umi_gdf.columns = ["height", "geometry", "template", "wwr"]

            return cls(
                gdf=umi_gdf,
                template_lib=template_lib,
                epw=epw,
                shoebox_width=width,
                floor_to_floor_height=f2f_height,
                perim_offset=perim_offset,
            )


if __name__ == "__main__":
    umi_test = Umi.open_uio(
        filename="D:/Users/zoelh/GitRepos/ml-for-building-energy-modeling/umi_data/Florianopolis/Florianopolis_Baseline.uio",
        epw_path="ml-for-bem/data/epws/global_epws_indexed/cityidx_0033_BRA_SP-São Paulo-Congonhas AP.837800_TRY Brazil.epw",
        archetypal_path="ml-for-bem/data/template_libs/cz_libs/residential/CZ3A.json",
    )
    print("SCHEDULES ARRAY: ", umi_test.schedules_array.shape)
    print("TEMPLATE DF: ", umi_test.features_df.shape)
    print("EPW ARRAY: ", umi_test.epw_array.shape)
    print(umi_test.gdf.head())
    umi_test.visualize_2d()

# TODO
# Shading vector
# mapping
