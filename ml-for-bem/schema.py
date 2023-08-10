import os
import json
from glob import glob

from functools import reduce
from pathlib import Path
from typing import List

import numpy as np
import math

import matplotlib.pyplot as plt
import logging

logging.basicConfig()
logger = logging.getLogger("Schema")
logger.setLevel(logging.INFO)

try:
    from archetypal import UmiTemplateLibrary
    from archetypal.idfclass.sql import Sql
    from archetypal.template.schedule import UmiSchedule
    from archetypal.template.materials.material_layer import MaterialLayer
    from archetypal.template.constructions.window_construction import WindowConstruction
    import pandas as pd
    from pyumi.shoeboxer.shoebox import ShoeBox
    from pyumi.epw import EPW
except (ImportError, ModuleNotFoundError) as e:
    logger.error("Failed to import a package! Be wary about continuing...", exc_info=e)


from schedules import (
    schedule_paths,
    operations,
    get_schedules,
    mutate_timeseries,
    update_schedule_objects,
)

from nrel_uitls import CLIMATEZONES_LIST, RESTYPES

data_path = Path(os.path.dirname(os.path.abspath(__file__))) / "data"

HIGH_LOW_MASS_THESH = 00000  # J/m2K


class ShoeboxConfiguration:
    """
    Stateful class for shoebox object args
    """

    __slots__ = (
        "width",
        "height",
        "facade_2_footprint",
        "perim_2_footprint",
        "roof_2_footprint",
        "footprint_2_ground",
        # "shading_fact",
        "wwr",
        "orientation",
    )

    def __init__(self):
        pass


class WhiteboxSimulation:
    """
    Class for configuring a whitebox simulation from a storage vector
    """

    __slots__ = (
        "schema",
        "storage_vector",
        "lib",
        "template",
        "epw_path",
        "shoebox_config",
        "shoebox",
        "hourly",
        "monthly",
        "epw",
    )
    JOULES_TO_KWH = 2.777e-7

    def __init__(self, schema, storage_vector):
        """
        Create a whitebox simulation object

        Args:
            schema: Schema, semantic method handler
            storage_vector: np.ndarray, shape=(len(storage_vector)), the storage vector to load
        Returns:
            A ready to simulate whitebox sim
        """
        self.schema = schema
        self.storage_vector = storage_vector
        self.shoebox_config = ShoeboxConfiguration()
        self.load_template()
        self.build_epw_path()
        self.update_parameters()
        self.build_shoebox()

    def load_template(self):
        """
        Method for loading a template based off id in storage vector.
        """
        template_lib_idx = self.schema["climate_zone"].extract_storage_values(
            self.storage_vector
        )

        template_lib_idx = int(template_lib_idx)

        template_lib_path = (
            data_path
            / "template_libs"
            / "cz_libs"
            / "residential"
            / f"CZ{CLIMATEZONES_LIST[template_lib_idx]}.json"
        )

        """PROG_{IDX}_VINTAGE_{IDX}_MASS_{0/1}"""

        vintage = self.schema["vintage"].extract_storage_values(self.storage_vector)

        vintage_idx = 0
        if vintage < 1940:
            pass
        elif vintage < 1980:
            vintage_idx = 1
        elif vintage < 2004:
            vintage_idx = 2
        else:
            vintage_idx = 3

        program_type = self.schema["program_type"].extract_storage_values(
            self.storage_vector
        )
        tmass = self.schema["FacadeMass"].extract_storage_values(self.storage_vector)

        mass_flag = 0
        if tmass > HIGH_LOW_MASS_THESH:
            mass_flag = 1

        n_programs = len(RESTYPES)
        n_masses = 2
        n_vintages = 4
        template_idx = (
            n_masses * n_vintages * int(program_type)
            + n_masses * vintage_idx
            + mass_flag
        )

        """
        0a - template library
            single family, pre-1940 low mass
            single family pe-1940 high mass
            multi family pre 1940
            multi big family pre 1940
            multi bigger family pre 1940
            single family pre 1980

        """
        self.lib = UmiTemplateLibrary.open(template_lib_path)
        self.template = self.lib.BuildingTemplates[template_idx]
        # print("Vintage", vintage, vintage_idx)
        # print("Program Type", int(program_type))
        # print("FacadeMass", tmass, high_mass)
        # print("Template Idx", template_idx)
        # for bt in self.lib.BuildingTemplates:
        #     print(bt.Name)

    def update_parameters(self):
        """
        Method for mutating semantic simulation objects
        """
        for parameter in self.schema.parameters:
            parameter.mutate_simulation_object(self)

    def build_epw_path(self):
        """
        Method for building the epw path
        """
        # TODO: improve this to use a specific map rather than a globber
        cityidx = self.schema["base_epw"].extract_storage_values(self.storage_vector)
        globber = (
            data_path / "epws" / "city_epws_indexed" / f"cityidx_{int(cityidx):04d}**"
        )
        files = glob(str(globber))
        self.epw_path = data_path / files[0]

    def load_epw(self):
        self.epw = EPW(self.epw_path)

    def build_shoebox(self):
        """
        Method for constructing the actual shoebox simulation object
        """
        wwr_map = {0: 0, 90: 0, 180: self.shoebox_config.wwr, 270: 0}  # N is 0, E is 90
        # Convert to coords
        width = self.shoebox_config.width
        depth = self.shoebox_config.height / self.shoebox_config.facade_2_footprint
        perim_depth = depth * self.shoebox_config.perim_2_footprint
        height = self.shoebox_config.height
        zones_data = [
            {
                "name": "Perim",
                "coordinates": [
                    (width, 0),
                    (width, perim_depth),
                    (0, perim_depth),
                    (0, 0),
                ],
                "height": height,
                "num_stories": 1,
                "zoning": "by_storey",
            },
            {
                "name": "Core",
                "coordinates": [
                    (width, perim_depth),
                    (width, depth),
                    (0, depth),
                    (0, perim_depth),
                ],
                "height": height,
                "num_stories": 1,
                "zoning": "by_storey",
            },
        ]

        sb = ShoeBox.from_template(
            building_template=self.template,
            zones_data=zones_data,
            wwr_map=wwr_map,
        )
        sb.epw = self.epw_path

        # Set floor and roof geometry for each zone
        for surface in sb.getsurfaces(surface_type="roof"):
            name = surface.Name
            name = name.replace("Roof", "Ceiling")
            sb.add_adiabatic_to_surface(
                surface, name, self.shoebox_config.roof_2_footprint
            )
        for surface in sb.getsurfaces(surface_type="floor"):
            name = surface.Name
            name = name.replace("Floor", "Int Floor")
            sb.add_adiabatic_to_surface(
                surface, name, self.shoebox_config.footprint_2_ground
            )
        # Make core walls adiabatic
        exterior_facade = sb.getsubsurfaces()[0].Building_Surface_Name
        for surface in sb.getsurfaces(surface_type="wall"):
            if ("Core").lower() in surface.Zone_Name.lower():
                surface.Outside_Boundary_Condition = "Adiabatic"
                surface.Sun_Exposure = "NoSun"
                surface.Wind_Exposure = "NoWind"
            if (
                "Perim"
            ).lower() in surface.Zone_Name.lower() and surface.Name is not exterior_facade:
                surface.Outside_Boundary_Condition = "Adiabatic"
                surface.Sun_Exposure = "NoSun"
                surface.Wind_Exposure = "NoWind"

        # Internal partition and glazing
        # Orientation

        # TODO: - confirm that these do not need to be moved inside of simulate parallel process
        outputs = [
            timeseries.to_output_dict() for timeseries in self.schema.timeseries_outputs
        ]
        sb.rotate(
            self.shoebox_config.orientation * 90
        )  # 0 is S facing windows, 90 is E facing windows
        for surface in sb.getsubsurfaces():
            if ("Core").lower() in surface.Building_Surface_Name.lower():
                sb.removeidfobject(surface)
        sb.outputs.add_custom(outputs)
        sb.outputs.apply()
        self.shoebox = sb

    def build_shading(self, heights, radius=10, override=True):
        """
        Adds shading objects to a shoebox idf according to a list of heights (m)
        of equal width divided between 180 degrees at a given radius (m).
        Using geomeppy def add_shading_block(*args, **kwargs)
        """
        if override:
            self.remove_shading_surfaces()

        # base point is at central bottom of shoebox - just in case shoebox is not centered at 0,0
        zones = self.shoebox.idfobjects["ZONE"]
        for zone in zones:
            for surface in zone.zonesurfaces:
                if (
                    surface.Surface_Type == "wall"
                    and surface.Outside_Boundary_Condition == "outdoors"
                ):
                    coords = surface.coords
                    facade_base_coords = []
                    for coord in coords:
                        if coord[2] == 0:
                            facade_base_coords.append(coord[:2])
        facade_base_coords = np.array(facade_base_coords)
        if len(facade_base_coords) > 2:
            raise TypeError("Multiple exterior walls found!")
        center_pt = np.mean(facade_base_coords, axis=0)

        # Get first point of shading
        vect = np.min(facade_base_coords, axis=0) - center_pt
        v_norm = vect / np.linalg.norm(vect)
        p_0 = v_norm * radius

        # make radial array
        count = len(heights)
        theta = math.pi / count

        rot_matrix = [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ]

        shading_coords = np.array([p_0])
        pt = np.reshape(p_0, (2, 1))
        for h in heights:
            pt = np.matmul(rot_matrix, pt)
            shading_coords = np.append(shading_coords, np.reshape(pt, (1, 2)), axis=0)
        shading_coords[:, 0] += center_pt[0]
        shading_coords[:, 1] += center_pt[1]

        for i, h in enumerate(heights):
            name = f"SHADER_{i}"
            base_coords = shading_coords[i : i + 2, :]
            coords = []
            coords.append((base_coords[0, 0], base_coords[0, 1], h))
            coords.append((base_coords[0, 0], base_coords[0, 1], 0.0))
            coords.append((base_coords[1, 0], base_coords[1, 1], 0.0))
            coords.append((base_coords[1, 0], base_coords[1, 1], h))
            shading_epbunch = self.shoebox.newidfobject(
                key="Shading:Building:Detailed".upper(),
                Name=name,
                Number_of_Vertices=4,
            )
            for j in coords:
                shading_epbunch.fieldvalues.extend(map(str, j))

    def remove_shading_surfaces(self):
        """
        Used to override existing shading surfces
        """
        for surface in self.shoebox.getshadingsurfaces():
            self.shoebox.removeidfobject(surface)

    def simulate(self):
        self.shoebox.simulate(verbose=False, prep_outputs=False, readvars=False)
        sql = Sql(self.shoebox.sql_file)
        series_to_retrieve_hourly = []
        series_to_retrieve_monthly = []
        for timeseries in self.schema.timeseries_outputs:
            if timeseries.store_output:
                if timeseries.freq.upper() == "MONTHLY":
                    series_to_retrieve_monthly.append(
                        timeseries.var_name
                        if timeseries.key_name is None
                        else timeseries.key_name
                    )
                if timeseries.freq.upper() == "HOURLY":
                    series_to_retrieve_hourly.append(
                        timeseries.var_name
                        if timeseries.key_name is None
                        else timeseries.key_name
                    )
        ep_df_hourly = pd.DataFrame(
            sql.timeseries_by_name(
                series_to_retrieve_hourly, reporting_frequency="Hourly"
            )
        )
        ep_df_monthly = pd.DataFrame(
            sql.timeseries_by_name(
                series_to_retrieve_monthly, reporting_frequency="Monthly"
            )
        )
        self.hourly = ep_df_hourly
        self.monthly = ep_df_monthly
        return ep_df_hourly, ep_df_monthly
        # ep_df_hourly_heating = pd.DataFrame(sql.timeseries_by_name("Zone Ideal Loads Zone Total Heating Energy", reporting_frequency="Hourly"))
        # ep_df_hourly_cooling = pd.DataFrame(sql.timeseries_by_name("Zone Ideal Loads Zone Total Cooling Energy", reporting_frequency="Hourly"))
        # ep_df_monthly_heating = pd.DataFrame(sql.timeseries_by_name("Zone Ideal Loads Zone Total Heating Energy", reporting_frequency="Monthly"))
        # ep_df_monthly_cooling = pd.DataFrame(sql.timeseries_by_name("Zone Ideal Loads Zone Total Cooling Energy", reporting_frequency="Monthly"))

    @property
    def totals(self):
        values = self.hourly.values
        aggregate = values.sum(axis=0) * self.JOULES_TO_KWH
        perim_heating = aggregate[0]
        perim_cooling = aggregate[1]
        core_heating = aggregate[2]
        core_cooling = aggregate[3]
        cooling = perim_cooling + core_cooling
        heating = perim_heating + core_heating
        return (heating, cooling), (
            heating / self.shoebox.total_building_area,
            cooling / self.shoebox.total_building_area,
        )

    def plot_results(self, start=0, length=8760, normalize=True, figsize=(10, 10)):
        if not hasattr(self, "epw"):
            self.load_epw()
        dbt = np.array(self.epw.dry_bulb_temperature.values)
        dbt_trimmed = dbt[start : start + length]
        dbt_trimmed_daily = np.mean(dbt_trimmed.reshape(-1, 24), axis=1).flatten()
        dbt_daily = self.epw.dry_bulb_temperature.average_daily()
        dbt_monthly = self.epw.dry_bulb_temperature.average_monthly()
        hourly = (
            self.hourly
            * self.JOULES_TO_KWH
            / (self.shoebox.total_building_area if normalize else 1)
        )
        hourly_trimmed = hourly[start : start + length]
        monthly = (
            self.monthly
            * self.JOULES_TO_KWH
            / (self.shoebox.total_building_area if normalize else 1)
        )
        lw = 2

        fig, axs = plt.subplots(5, 1, figsize=figsize)
        axs[0].plot(
            hourly["System"]["BLOCK PERIM STOREY 0 IDEAL LOADS AIR SYSTEM"]
            .resample("1D")
            .sum(),
            linewidth=lw,
            label=["Perim-Heating", "Perim-Cooling"],
        )
        axs[0].plot(
            hourly["System"]["BLOCK CORE STOREY 0 IDEAL LOADS AIR SYSTEM"]
            .resample("1D")
            .sum(),
            linewidth=lw,
            label=["Core-Heating", "Core-Cooling"],
        )
        axs[0].set_title("Daily")
        axs[0].set_ylabel(f"kWhr{'/m2' if normalize else ''}")
        axs[0].legend()
        axs_0b = axs[0].twinx()
        axs_0b.plot(
            hourly.resample("1D").mean().index,
            dbt_daily,
            linewidth=lw / 2,
            label="Temp",
        )
        axs_0b.legend()
        axs_0b.set_ylabel("deg. C")

        axs[1].plot(
            hourly_trimmed["System"]["BLOCK PERIM STOREY 0 IDEAL LOADS AIR SYSTEM"],
            linewidth=lw,
            label=["Perim-Heating", "Perim-Cooling"],
        )
        axs[1].plot(
            hourly_trimmed["System"]["BLOCK CORE STOREY 0 IDEAL LOADS AIR SYSTEM"],
            linewidth=lw,
            label=["Core-Heating", "Core-Cooling"],
        )
        axs[1].set_title("Hourly")
        axs[1].set_ylabel(f"kWhr{'/m2' if normalize else ''}")
        axs[1].legend()
        axs_1b = axs[1].twinx()
        axs_1b.plot(
            hourly_trimmed.index,
            dbt[start : start + length],
            linewidth=lw / 2,
            label="Temp",
        )
        axs_1b.legend()
        axs_1b.set_ylabel("deg. C")

        axs[2].plot(
            hourly_trimmed["System"]["BLOCK PERIM STOREY 0 IDEAL LOADS AIR SYSTEM"]
            .resample("1D")
            .sum(),
            linewidth=lw,
            label=["Perim-Heating", "Perim-Cooling"],
        )
        axs[2].plot(
            hourly_trimmed["System"]["BLOCK CORE STOREY 0 IDEAL LOADS AIR SYSTEM"]
            .resample("1D")
            .sum(),
            linewidth=lw,
            label=["Core-Heating", "Core-Cooling"],
        )
        axs[2].set_title("Daily")
        axs[2].set_ylabel(f"kWhr{'/m2' if normalize else ''}")
        axs[2].legend()
        axs_2b = axs[2].twinx()
        axs_2b.plot(
            hourly_trimmed.resample("1D").mean().index,
            dbt_trimmed_daily,
            linewidth=lw / 2,
            label="Temp",
        )
        axs_2b.legend()
        axs_2b.set_ylabel("deg. C")

        axs[3].plot(
            monthly["System"]["BLOCK PERIM STOREY 0 IDEAL LOADS AIR SYSTEM"],
            linewidth=lw,
            label=["Perim-Heating", "Perim-Cooling"],
        )
        axs[3].plot(
            monthly["System"]["BLOCK CORE STOREY 0 IDEAL LOADS AIR SYSTEM"],
            linewidth=lw,
            label=["Core-Heating", "Core-Cooling"],
        )
        axs[3].set_title("Monthly")
        axs[3].set_ylabel(f"kWhr{'/m2' if normalize else ''}")
        axs[3].legend()
        axs_3b = axs[3].twinx()
        axs_3b.plot(monthly.index, dbt_monthly, linewidth=lw / 2, label="Temp")
        axs_3b.legend()
        axs_3b.set_ylabel("deg. C")

        daily = hourly["System"].resample("1D").sum().values
        axs[4].scatter(dbt_daily, daily[:, 0], s=lw, label="Perim Heating")
        axs[4].scatter(dbt_daily, daily[:, 1], s=lw, label="Perim Cooling")
        axs[4].scatter(dbt_daily, daily[:, 2], s=lw, label="Core Heating")
        axs[4].scatter(dbt_daily, daily[:, 3], s=lw, label="Core Cooling")
        axs[4].set_title("Load vs Daily Temp")
        axs[4].legend()
        axs[4].set_ylabel(f"kWhr{'/m2' if normalize else ''}")
        axs[4].set_xlabel("deg. C")
        plt.tight_layout(pad=1)

    def summarize(self):
        print("\n\n" + "-" * 30)
        print("EPW:", self.epw_path)
        print("Selected Template:", self.template.Name)
        print("---ShoeboxConfig---")
        print("Height", self.shoebox_config.height)
        print("Width", self.shoebox_config.width)
        print("WWR", self.shoebox_config.wwr)
        print("Facade2Foot", self.shoebox_config.facade_2_footprint)
        print("Perim2Foot", self.shoebox_config.perim_2_footprint)
        print("Foot2Gnd [adia %]", self.shoebox_config.footprint_2_ground)
        print("Roof2Gnd [adia %]", self.shoebox_config.roof_2_footprint)
        print("Orientation", self.shoebox_config.orientation)
        print("---PERIM/CORE Values---")
        print(
            "Heating Setpoint:",
            self.template.Perimeter.Conditioning.HeatingSetpoint,
            self.template.Core.Conditioning.HeatingSetpoint,
        )
        print(
            "Cooling Setpoint:",
            self.template.Perimeter.Conditioning.CoolingSetpoint,
            self.template.Core.Conditioning.CoolingSetpoint,
        )
        print(
            "Equipment Power Density:",
            self.template.Perimeter.Loads.EquipmentPowerDensity,
            self.template.Core.Loads.EquipmentPowerDensity,
        )
        print(
            "Lighting Power Density:",
            self.template.Perimeter.Loads.LightingPowerDensity,
            self.template.Core.Loads.LightingPowerDensity,
        )
        print(
            "People Density:",
            self.template.Perimeter.Loads.PeopleDensity,
            self.template.Core.Loads.PeopleDensity,
        )
        print(
            "Infiltration:",
            self.template.Perimeter.Ventilation.Infiltration,
            self.template.Core.Ventilation.Infiltration,
        )
        print(
            "Roof HeatCap:",
            self.template.Perimeter.Constructions.Roof.heat_capacity_per_unit_wall_area,
            self.template.Core.Constructions.Roof.heat_capacity_per_unit_wall_area,
        )
        print(
            "Facade HeatCap:",
            self.template.Perimeter.Constructions.Facade.heat_capacity_per_unit_wall_area,
            self.template.Core.Constructions.Facade.heat_capacity_per_unit_wall_area,
        )
        print(
            "U Window:", self.template.Windows.Construction.u_value
        )  # TODO: this is slightly different!)
        print(
            "VLT",
            self.template.Windows.Construction.Layers[0].Material.VisibleTransmittance,
        )
        print("Roof RSI:", self.template.Perimeter.Constructions.Roof.r_value)
        print("Facade RSI:", self.template.Perimeter.Constructions.Facade.r_value)
        print("Slab RSI:", self.template.Perimeter.Constructions.Slab.r_value)
        print("Partition RSI:", self.template.Perimeter.Constructions.Partition.r_value)
        print("Ground RSI:", self.template.Perimeter.Constructions.Ground.r_value)
        print("Roof Assembly:", self.template.Perimeter.Constructions.Roof.Layers)
        print("Facade Assembly:", self.template.Perimeter.Constructions.Facade.Layers)
        print(
            "Partition Assembly:",
            self.template.Perimeter.Constructions.Partition.Layers,
        )
        print("Slab Assembly:", self.template.Perimeter.Constructions.Slab.Layers)
        print("Window Assembly:", self.template.Windows.Construction.Layers)


class SchemaParameter:
    """
    Base class for semantically representing operations on numpy/torch tensors
    which handles mutations of storage vectors, methods for updating simulation objects,
    and generating ML vectors from storage vectors, etc
    """

    __slots__ = (
        "name",
        "dtype",
        "start_storage",
        "start_ml",
        "shape_storage",
        "shape_ml",
        "len_storage",
        "len_ml",
        "in_ml",
        "info",
        "source",
    )

    def __init__(
        self,
        name,
        info,
        source=None,
        shape_storage=(1,),
        shape_ml=(1,),
        dtype="scalar",
    ):
        self.name = name
        self.info = info
        self.source = source
        self.dtype = dtype

        self.shape_storage = shape_storage
        self.shape_ml = shape_ml
        if shape_ml == (0,):
            self.in_ml = False
        else:
            self.in_ml = True

        self.len_storage = reduce(lambda a, b: a * b, shape_storage)
        self.len_ml = reduce(lambda a, b: a * b, shape_ml)

    def __repr__(self):
        return f"---{self.name}---\nshape_storage={self.shape_storage}, shape_ml={self.shape_ml}, dtype={self.dtype}\n{self.info}"

    def extract_storage_values(self, storage_vector):
        """
        Extract data values for this parameter from the current storage vector.  If this parameter represents matrix data,
        the data will be reshaped into the appropriate shape.
        Args:
            storage_vector: np.ndarray, shape=(len(storage_vector)) to extract data from
        Returns:
            data: float or np.ndarray, shape=(*parameter.shape), data associated with this parameter
        """
        data = storage_vector[
            self.start_storage : self.start_storage + self.len_storage
        ]
        if self.shape_storage == (1,):
            return data[0]
        else:
            return data.reshape(*self.shape_storage)

    def extract_storage_values_batch(self, storage_batch):
        """
        Extract data values for this parameter from all vectors in a storage batch.  If this parameter represents matrix data,
        the data will be reshaped into the appropriate shape so possibly a tensor if the parameter stores matrix data).
        Args:
            storage_batch: np.ndarray, shape=(n_vectors_in_batch, len(storage_vector)) to extract data from
        Returns:
            data: np.ndarray, shape=(n_vectors_in_batch, *parameter.shape), data associated with this parameter for each vector in batch
        """
        data = storage_batch[
            :, self.start_storage : self.start_storage + self.len_storage
        ]
        return data.reshape(-1, *self.shape_storage)

    def to_ml(self, storage_batch=None, value=None):
        if not self.in_ml:
            logger.warning(
                f"Attempted to call 'SchemaParameter.to_ml(storage_batch)' on PARAMETER:{self.name} but that parameter is not included in the ML vector.  You can ignore this message."
            )
        else:
            if isinstance(self, OneHotParameter):
                counts = (
                    self.extract_storage_values_batch(storage_batch)
                    if value is None
                    else value
                )
                onehots = np.zeros((counts.shape[0], self.count))
                onehots[np.arange(counts.shape[0]), counts[:, 0].astype(int)] = 1
                return onehots
            elif isinstance(self, SchedulesParameters):
                return (
                    self.extract_storage_values_batch(storage_batch)
                    if value is None
                    else value
                )
            elif isinstance(self, WindowParameter):
                vals = self.normalize(
                    self.extract_storage_values_batch(storage_batch).reshape(
                        -1, *self.shape_ml
                    )
                    if value is None
                    else value
                )
                return vals
            else:
                vals = self.normalize(
                    self.extract_storage_values_batch(storage_batch).reshape(
                        -1, *self.shape_ml
                    )
                    if value == None
                    else value
                )
                return vals

    def normalize(self, val):
        """
        Normalize data according to the model's schema.  For base SchemaParameters, this method
        does nothing.  Descendents of this (e.g. numerics) which require normalization implement
        their own methods for normalization.
        Args:
            val: np.ndarray, data to normalize
        Returns:
            val: np.ndarray, normalized data
        """
        return val

    def unnormalize(self, val):
        """
        Unnormalize data according to the model's schema.  For base SchemaParameters, this method
        does nothing.  Descendents of this (e.g. numerics) which require normalization implement
        their own methods for unnormalization.
        Args:
            val: np.ndarray, data to unnormalize
        Returns:
            val: np.ndarray, unnormalized data
        """
        return val

    def clip(self, val):
        """
        Clip values to the input range if defined.
        This method does nothing, descendents should implement if needed.
        Args:
            val: np.ndarray, data to clip
        Returns:
            val: np.ndarray, clipped data
        """
        return val

    def mutate_simulation_object(self, whitebox_sim: WhiteboxSimulation):
        """
        This method updates the simulation objects (archetypal template, shoebox config)
        by extracting values for this parameter from the sim's storage vector and using this
        parameter's logic to update the appropriate objects.
        The default base SchemaParameter does nothing.  Children classes implement the appropriate
        semantic logic.
        Args:
            whitebox_sim: WhiteboxSimulation
        """
        pass

    def extract_from_template(self):
        """
        This method extracts the parameter value from an archetypal building template for the creation of a building vector.
        Works as the reverse of mutate_simulation_object
        Args:
            whitebox_sim: WhiteboxSimulation
            building_template: Archetypal BuildingTemplate #TODO: should the building template be a parameter of the whitebox object?
        """
        pass


class NumericParameter(SchemaParameter):
    """
    Numeric parameters which have mins/maxs/ranges can inherit this class in order
    to gain the ability to normalize/unnormalize
    """

    __slots__ = ("min", "max", "range", "mean", "std")

    def __init__(self, min=0, max=1, mean=0.5, std=0.25, **kwargs):
        super().__init__(**kwargs)
        self.min = min
        self.max = max
        self.mean = mean
        self.std = std
        self.range = self.max - self.min

    def normalize(self, value):
        return (value - self.min) / self.range

    def unnormalize(self, value):
        # TODO: test that norm works for windows which are a vector
        return value * self.range + self.min

    def clip(self, val):
        """
        Clip values to the input range if defined.
        This method does nothing, descendents should implement if needed.
        Args:
            val: np.ndarray, data to clip
        Returns:
            val: np.ndarray, clipped data
        """
        # TODO: test that clip works for windows which are a vector
        return np.clip(val, self.min, self.max)


class OneHotParameter(SchemaParameter):
    __slots__ = "count"

    def __init__(self, count, shape_ml=None, **kwargs):
        super().__init__(
            dtype="onehot",
            shape_ml=(count,) if shape_ml == None else shape_ml,
            **kwargs,
        )
        self.count = count


class ShoeboxGeometryParameter(NumericParameter):
    __slots__ = ()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def mutate_simulation_object(self, whitebox_sim: WhiteboxSimulation):
        """
        This method updates the simulation objects (archetypal template, shoebox config)
        by extracting values for this parameter from the sim's storage vector and using this
        parameter's logic to update the appropriate objects.
        Updates whitebox simulation's shoebox configuration dictionary class.
        Args:
            whitebox_sim: WhiteboxSimulation
        """
        value = self.extract_storage_values(whitebox_sim.storage_vector)
        setattr(whitebox_sim.shoebox_config, self.name, value)

    def extract_from_template(self, whitebox_sim=None):
        """
        This method extracts the parameter value from an archetypal building template for the creation of a building vector.
        Works as the reverse of mutate_simulation_object
        Args:
            whitebox_sim: WhiteboxSimulation
            building_template: Archetypal BuildingTemplate #TODO: should the building template be a parameter of the whitebox object?
        """
        if whitebox_sim:
            return self.to_ml(value=getattr(whitebox_sim.shoebox_config, self.name))
        else:
            return self.to_ml(value=self.mean)


class ShoeboxOrientationParameter(OneHotParameter):
    __slots__ = ()

    def __init__(self, **kwargs):
        super().__init__(count=4, **kwargs)

    def mutate_simulation_object(self, whitebox_sim: WhiteboxSimulation):
        value = self.extract_storage_values(whitebox_sim.storage_vector)
        setattr(whitebox_sim.shoebox_config, self.name, value)


class BuildingTemplateParameter(NumericParameter):
    __slots__ = "path"

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = path.split(".")

    def mutate_simulation_object(self, whitebox_sim: WhiteboxSimulation):
        """
        This method updates the simulation objects (archetypal template, shoebox config)
        by extracting values for this parameter from the sim's storage vector and using this
        parameter's logic to update the appropriate objects.
        Updates whitebox simulation's direct building template parameters.

        Args:
            whitebox_sim: WhiteboxSimulation
        """
        value = self.extract_storage_values(whitebox_sim.storage_vector)
        template_param = self.path[-1]
        for zone in ["Perimeter", "Core"]:
            path = [whitebox_sim.template, zone, *self.path]
            path = path[:-1]
            object_to_update = reduce(lambda a, b: a[b], path)
            setattr(object_to_update, template_param, value)

    def extract_from_template(self, building_template):
        """
        This method extracts the parameter value from an archetypal building template for the creation of a building vector.
        Works as the reverse of mutate_simulation_object
        Args:
            whitebox_sim: WhiteboxSimulation
            building_template: Archetypal BuildingTemplate
        """
        val = building_template.Perimeter
        for attr in self.path:
            val = getattr(val, attr)
        return self.to_ml(value=val)


class RValueParameter(BuildingTemplateParameter):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

    def mutate_simulation_object(self, whitebox_sim: WhiteboxSimulation):
        """
        Thi updates the simulation objects (archetypal template, shoebox config)
        by extracting values for this parameter from the sim's storage vector and using this
        parameter's logic to update the appropriate objects.
        Updates whitebox simulation's r value parameter by inferring the insulation layer and updating its
        thickness automaticaly.

        Args:
            whitebox_sim: WhiteboxSimulation
        """
        desired_r_value = self.extract_storage_values(whitebox_sim.storage_vector)
        for zone in ["Perimeter", "Core"]:
            zone_obj = getattr(whitebox_sim.template, zone)
            constructions = zone_obj.Constructions
            construction = getattr(constructions, self.path[0])
            # TODO: make sure units are correct!!!
            # = self.infer_insulation_layer()
            layers = construction.Layers
            insulation_layer_ix = None
            k_min = 999999
            for i, layer in enumerate(layers):
                if layer.Material.Conductivity < k_min:
                    k_min = layer.Material.Conductivity
                    insulation_layer_ix = i

            i = insulation_layer_ix
            all_layers_except_insulation_layer = [a for a in layers]
            all_layers_except_insulation_layer.pop(i)
            insulation_layer: MaterialLayer = layers[i]

            if desired_r_value <= sum(
                [a.r_value for a in all_layers_except_insulation_layer]
            ):
                raise ValueError(
                    f"Cannot set assembly r-value smaller than "
                    f"{sum([a.r_value for a in all_layers_except_insulation_layer])} "
                    f"because it would result in an insulation of a "
                    f"negative thickness. Try a higher value or changing the material "
                    f"layers instead."
                )

            alpha = float(desired_r_value) / construction.r_value
            new_r_value = (
                (
                    (alpha - 1)
                    * sum([a.r_value for a in all_layers_except_insulation_layer])
                )
            ) + alpha * insulation_layer.r_value
            insulation_layer.r_value = new_r_value
            if insulation_layer.Thickness <= 0.003:
                construction.Layers = all_layers_except_insulation_layer

    def extract_from_template(self, building_template):
        """
        This method extracts the parameter value from an archetypal building template for the creation of a building vector.
        Works as the reverse of mutate_simulation_object
        Args:
            whitebox_sim: WhiteboxSimulation
            building_template: Archetypal BuildingTemplate #TODO: should the building template be a parameter of the whitebox object?
        """
        if "Facade" in self.name:
            surface = "Facade"
        elif "Roof" in self.name:
            surface = "Roof"
        elif "Slab" in self.name:
            surface = "Slab"
        path = ["Constructions", surface]
        val = building_template.Perimeter
        for attr in path:
            val = getattr(val, attr)
        return self.to_ml(value=val.r_value)


class TMassParameter(BuildingTemplateParameter):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

    def mutate_simulation_object(self, whitebox_sim: WhiteboxSimulation):
        desired_heat_capacity_per_wall_area = self.extract_storage_values(
            whitebox_sim.storage_vector
        )
        for zone in ["Perimeter", "Core"]:
            zone_obj = getattr(whitebox_sim.template, zone)
            constructions = zone_obj.Constructions
            construction = getattr(constructions, self.path[0])

            concrete_layer = construction.Layers[0]  # concrete
            material = concrete_layer.Material
            cp = material.SpecificHeat
            rho = material.Density
            volumetric_cp = cp * rho
            old_thermal_mass = construction.heat_capacity_per_unit_wall_area
            thermal_mass_without_concrete = (
                old_thermal_mass - concrete_layer.heat_capacity
            )
            thickness = (
                desired_heat_capacity_per_wall_area - thermal_mass_without_concrete
            ) / volumetric_cp
            # thickness = desired_heat_capacity_per_wall_area / (cp * rho)
            if thickness < 0.004:
                all_layers_except_mass = [a for a in construction.Layers]
                all_layers_except_mass.pop(0)
                construction.Layers = all_layers_except_mass
            else:
                concrete_layer.Thickness = thickness

    def extract_from_template(self, building_template):
        """
        This method extracts the parameter value from an archetypal building template for the creation of a building vector.
        Works as the reverse of mutate_simulation_object
        Args:
            whitebox_sim: WhiteboxSimulation
            building_template: Archetypal BuildingTemplate #TODO: should the building template be a parameter of the whitebox object?
        """
        zone = "Perimeter"
        if "Facade" in self.name:
            surface = "Facade"
        elif "Roof" in self.name:
            surface = "Roof"
        elif "Slab" in self.name:
            surface = "Slab"
        path = ["Constructions", surface, "heat_capacity_per_unit_wall_area"]
        val = building_template.Perimeter
        for attr in path:
            val = getattr(val, attr)
        return self.to_ml(value=val)


class WindowParameter(NumericParameter):
    def __init__(self, min, max, **kwargs):
        super().__init__(shape_storage=(3,), shape_ml=(3,), **kwargs)
        self.min = np.array(min)
        self.max = np.array(max)
        self.range = self.max - self.min

    def mutate_simulation_object(self, whitebox_sim: WhiteboxSimulation):
        """
        This method updates the simulation objects (archetypal template, shoebox config)
        by extracting values for this parameter from the sim's storage vector and using this
        parameter's logic to update the appropriate objects.
        Updates whitebox simulation's r value parameter by inferring the insulation layer and updating its
        thickness automaticaly.

        Args:
            whitebox_sim: WhiteboxSimulation
        """
        # Get the var id and batch id for naming purposes
        variation_id = whitebox_sim.schema["variation_id"].extract_storage_values(
            whitebox_sim.storage_vector
        )
        batch_id = whitebox_sim.schema["batch_id"].extract_storage_values(
            whitebox_sim.storage_vector
        )

        # get the three values for u/shgc/vlt
        values = self.extract_storage_values(whitebox_sim.storage_vector)

        # separate them
        u_value = values[0]
        shgc = values[1]
        vlt = values[2]

        # create a new single layer window that has the properties from special single layer material
        window = WindowConstruction.from_shgc(
            Name=f"window-{int(batch_id):05d}-{int(variation_id):05d}",
            solar_heat_gain_coefficient=shgc,
            u_factor=u_value,
            visible_transmittance=vlt,
        )

        # Update the window
        whitebox_sim.template.Windows.Construction = window

    def extract_from_template(self, building_template):
        """
        This method extracts the parameter value from an archetypal building template for the creation of a building vector.
        Works as the reverse of mutate_simulation_object
        Args:
            whitebox_sim: WhiteboxSimulation
            building_template: Archetypal BuildingTemplate #TODO: should the building template be a parameter of the whitebox object?
        """
        # return self.to_ml(
        #     value=np.array(
        #         [
        #             building_template.Windows.Construction.u_value,
        #             0.5,  # TODO SHGC
        #             building_template.Windows.Construction.visible_transmittance,
        #         ]
        #     )
        # )
        window = building_template.Windows.Construction
        uval = window.u_value
        if window.glazing_count == 2:
            shgc = window.shgc()
        elif window.glazing_count == 1:
            # Calculate shgc from t_sol of construction
            tsol = window.Layers[0].Material.SolarTransmittance
            shgc = self.single_pane_shgc_estimation(tsol, uval)
        else:
            # if window is not 2 layers
            logging.info(
                f"Window is {window.glazing_count} layers. Assuming SHGC is 0.6"
            )
            shgc = 0.6

        window_array = np.array(
            [
                uval,
                shgc,  # TODO SHGC
                window.visible_transmittance,
            ]
        )
        # return self.normalize(value=window_array)
        # window_array = np.reshape(window_array,)
        return self.to_ml(value=window_array)
        # return window_array

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
                logger.info(
                    "WARNING: could not calculate shgc - review window parameters. Defaulting to 0.6."
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

    # def normalize(self, window_array):
    #     # TODO: discuss this - normalizing only the u_value
    #     print("NORMALIZING WINDOW ARRAY")
    #     norm_window_array = window_array.copy()
    #     print(f"Min {self.min[0]}, max {self.max[0]}, range{self.range[0]}")
    #     norm_val = (window_array[0] - self.min[0]) / self.range[0]
    #     print(norm_val)
    #     norm_window_array[0] = norm_val
    #     print(window_array, norm_window_array)
    #     return norm_window_array


class SchedulesParameters(SchemaParameter):
    __slots__ = ()
    paths = schedule_paths
    operations = operations
    op_indices = {operation: ix for ix, operation in enumerate(operations)}

    def __init__(self, **kwargs):
        super().__init__(
            name="schedules",
            dtype="matrix",
            shape_storage=(len(self.paths), len(self.operations)),
            shape_ml=(len(self.paths), 8760),
            **kwargs,
        )

    def mutate_simulation_object(self, whitebox_sim: WhiteboxSimulation):
        """
        Mutate a template's schedules according to a deterministic sequence of operations stored in the
        storage vector

        Args:
            whitebox_sim (WhiteboxSimulation): the simulation object with template to configure.
        """
        # TODO: avoid double mutation of recycled schedule - i think this is fixed, should confirm.
        seed = int(
            whitebox_sim.schema["schedules_seed"].extract_storage_values(
                whitebox_sim.storage_vector
            )
        )
        schedules = get_schedules(
            whitebox_sim.template, zones=["Core"], paths=self.paths
        )
        operations_map = self.extract_storage_values(whitebox_sim.storage_vector)
        new_schedules = mutate_timeseries(schedules, operations_map, seed)
        update_schedule_objects(
            whitebox_sim.template,
            timeseries=new_schedules,
            zones=["Core"],
            paths=self.paths,
            id=seed,
        )
        update_schedule_objects(
            whitebox_sim.template,
            timeseries=new_schedules,
            zones=["Perimeter"],
            paths=self.paths,
            id=seed,
        )
        whitebox_sim.template.Perimeter.Conditioning.MechVentSchedule = (
            whitebox_sim.template.Perimeter.Loads.OccupancySchedule
        )
        whitebox_sim.template.Perimeter.DomesticHotWater.WaterSchedule = (
            whitebox_sim.template.Perimeter.Loads.OccupancySchedule
        )

    def extract_from_template(self, building_template):
        """
        This method extracts the parameter value from an archetypal building template for the creation of a building vector.
        Works as the reverse of mutate_simulation_object
        Args:
            whitebox_sim: WhiteboxSimulation
            building_template: Archetypal BuildingTemplate #TODO: should the building template be a parameter of the whitebox object?
        """
        schedules = get_schedules(building_template, zones=["Core"], paths=self.paths)
        return self.to_ml(value=schedules)


class TimeSeriesOutput:
    __slots__ = (
        "name",
        "var_name",
        "key_name",
        "freq",
        "key",
        "store_output",
    )

    def __init__(
        self,
        name,
        var_name=None,
        key_name=None,
        store_output=True,
        freq="hourly",
        key="OUTPUT:VARIABLE",
    ):
        self.name = name
        self.var_name = var_name
        self.key_name = key_name
        self.freq = freq
        self.key = key
        self.store_output = store_output

    def to_output_dict(self):
        ep_dict = dict(
            key=self.key,
            Reporting_Frequency=self.freq,
        )
        if self.var_name:
            ep_dict["Variable_Name"] = self.var_name
        if self.key_name:
            ep_dict["Key_Name"] = self.key_name
        return ep_dict


class Schema:
    __slots__ = (
        "parameters",
        "timeseries_outputs",
        "storage_vec_len",
        "ml_vec_len",
        "_key_ix_lookup",
        "sim_output_configs",
        "sim_output_shape",
    )

    def __init__(
        self,
        parameters: List[SchemaParameter] = None,
        timeseries_outputs: List[TimeSeriesOutput] = None,
    ):
        if parameters != None:
            self.parameters = parameters
        else:
            self.parameters = [
                SchemaParameter(
                    name="batch_id",
                    dtype="index",
                    shape_ml=(0,),
                    info="batch_id of design",
                ),
                SchemaParameter(
                    name="variation_id",
                    dtype="index",
                    shape_ml=(0,),
                    info="variation_id of design",
                ),
                OneHotParameter(
                    name="program_type",
                    count=19,
                    info="Indicator of program type",
                    shape_ml=(0,),
                ),
                NumericParameter(
                    name="vintage",
                    info="The year of construction",
                    min=1920,
                    max=2020,
                    mean=1980,
                    std=20,
                    shape_ml=(0,),
                ),
                OneHotParameter(
                    name="climate_zone",
                    count=17,
                    info="Lookup index of template library to use.",
                    shape_ml=(0,),
                ),
                SchemaParameter(
                    name="base_epw",
                    dtype="index",
                    shape_ml=(0,),
                    info="Lookup index of EPW file to use.",
                ),
                ShoeboxGeometryParameter(
                    name="width",
                    min=3,
                    max=12,
                    mean=5,
                    std=1,
                    source="battini_shoeboxing_2023",
                    info="Width [m]",
                ),
                ShoeboxGeometryParameter(
                    name="height",
                    min=2.5,
                    max=6,
                    mean=3,
                    std=0.5,
                    source="ComStock",
                    info="Height [m]",
                ),
                ShoeboxGeometryParameter(
                    name="facade_2_footprint",
                    min=0.25,
                    max=1.5,
                    mean=0.75,
                    std=0.15,
                    source="dogan_shoeboxer_2017",
                    info="Facade to footprint ratio (unitless)",
                ),
                ShoeboxGeometryParameter(
                    name="perim_2_footprint",
                    min=0.05,
                    max=0.95,
                    mean=0.5,
                    std=0.25,
                    source="dogan_shoeboxer_2017",
                    info="Perimeter to footprint ratio (unitless)",
                ),
                ShoeboxGeometryParameter(
                    name="roof_2_footprint",
                    min=0.05,
                    max=0.95,
                    mean=0.5,
                    std=0.25,
                    source="dogan_shoeboxer_2017",
                    info="Roof to footprint ratio (unitless)",
                ),
                ShoeboxGeometryParameter(
                    name="footprint_2_ground",  # TODO: change this to ground_2_footprint for cohesiveness
                    min=0.05,
                    max=0.95,
                    mean=0.5,
                    std=0.25,
                    source="dogan_shoeboxer_2017",
                    info="Footprint to ground ratio (unitless)",
                ),
                # ShoeboxGeometryParameter(
                #     name="shading_fact",
                #     min=0,
                #     max=1,
                #     mean=0.1,
                #     std=0.33,
                #     info="Shading fact (unitless)",
                # ),
                ShoeboxGeometryParameter(
                    name="wwr",
                    min=0.05,
                    max=0.9,
                    mean=0.3,
                    std=0.25,
                    info="Window-to-wall Ratio (unitless)",
                ),
                ShoeboxOrientationParameter(
                    name="orientation",
                    info="Shoebox Orientation",
                ),
                BuildingTemplateParameter(
                    name="HeatingSetpoint",
                    path="Conditioning.HeatingSetpoint",
                    min=14,
                    max=30,
                    mean=21,
                    std=4,
                    info="Heating setpoint",
                ),
                BuildingTemplateParameter(
                    name="CoolingSetpoint",
                    path="Conditioning.CoolingSetpoint",
                    min=14,
                    max=30,
                    mean=22,
                    std=4,
                    info="Cooling setpoint",
                ),
                # BuildingTemplateParameter(
                #     name="HeatingCoeffOfPerf",
                #     path="Conditioning.HeatingCoeffOfPerf",
                #     min=0.9,
                #     max=5,
                #     mean=1,
                #     std=1,
                #     source="tacit",
                #     info="heating cop",
                # ),
                # BuildingTemplateParameter(
                #     name="CoolingCoeffOfPerf",
                #     path="Conditioning.CoolingCoeffOfPerf",
                #     min=2.5,
                #     max=5,
                #     mean=3,
                #     std=1,
                #     source="tacit",
                #     info="cooling cop",
                # ),
                BuildingTemplateParameter(
                    name="LightingPowerDensity",
                    path="Loads.LightingPowerDensity",
                    min=0,
                    max=20,
                    mean=10,
                    std=6,
                    source="ComStock",
                    info="Lighting Power Density [W/m2]",
                ),
                BuildingTemplateParameter(
                    name="EquipmentPowerDensity",
                    path="Loads.EquipmentPowerDensity",
                    min=0.1,
                    max=30,  # TODO this is foor super high density spaces (like mech rooms). Alternative is 500
                    mean=10,
                    std=6,
                    source="ComStock",
                    info="Equipment Power Density [W/m2]",
                ),
                BuildingTemplateParameter(
                    name="PeopleDensity",
                    path="Loads.PeopleDensity",
                    min=0,
                    max=0.5,
                    mean=0.1,
                    std=0.1,
                    source="ComStock",
                    info="People Density [people/m2]",
                ),
                BuildingTemplateParameter(
                    name="Infiltration",
                    path="Ventilation.Infiltration",
                    min=0.1,
                    max=4,
                    mean=2,
                    std=1,
                    source="tacit",
                    info="Infiltration rate [ach]",
                ),
                TMassParameter(
                    name="FacadeMass",
                    path="Facade",
                    min=13000,
                    max=800000,
                    mean=80000,
                    std=20000,
                    source="https://www.designingbuildings.co.uk/",
                    info="Exterior wall thermal mass (J/Km2)",
                ),
                TMassParameter(
                    name="RoofMass",
                    path="Roof",
                    min=13000,
                    max=1500000,
                    mean=200000,
                    std=200000,
                    source="https://www.designingbuildings.co.uk/",
                    info="Exterior roof thermal mass (J/Km2)",
                ),
                RValueParameter(
                    name="FacadeRValue",
                    path="Facade",
                    min=0.3,
                    max=15,
                    mean=2,
                    std=1,
                    source="ComStock, tacit knowledge",
                    info="Facade R-value",
                ),
                RValueParameter(
                    name="RoofRValue",
                    path="Roof",
                    min=0.3,
                    max=18,
                    mean=3.5,
                    std=1,
                    source="ComStock, tacit knowledge",
                    info="Roof R-value",
                ),
                RValueParameter(
                    name="SlabRValue",
                    path="Slab",
                    min=0.3,
                    max=15,
                    mean=2,
                    std=1,
                    source="ComStock, tacit knowledge",
                    info="Slab R-value",
                ),
                WindowParameter(
                    name="WindowSettings",
                    min=(0.3, 0.05, 0.05),
                    max=(7.0, 0.99, 0.99),
                    mean=np.array([5, 0.5, 0.5]),
                    std=np.array([2, 0.1, 0.1]),
                    source="climate studio",
                    info="U-value (m2K/W), shgc, vlt",
                ),
                SchemaParameter(
                    name="schedules_seed",
                    shape_ml=(0,),
                    dtype="index",
                    info="A seed to reliably reproduce schedules from the storage vector's schedule operations when generating ml vector",
                ),
                SchedulesParameters(
                    info="A matrix in the storage vector with operations to apply to schedules; a matrix of timeseries in ml vector",
                ),
            ]

        self.timeseries_outputs = [
            TimeSeriesOutput(
                name="Heating",
                key="OUTPUT:VARIABLE",
                var_name="Zone Ideal Loads Zone Total Heating Energy",
                freq="hourly",
                store_output=True,
            ),
            TimeSeriesOutput(
                name="Cooling",
                key="OUTPUT:VARIABLE",
                var_name="Zone Ideal Loads Zone Total Cooling Energy",
                freq="hourly",
                store_output=True,
            ),
            TimeSeriesOutput(
                name="Lighting",
                key="OUTPUT:VARIABLE",
                var_name="Lights Total Heating Energy",
                freq="hourly",
                store_output=False,
            ),
            TimeSeriesOutput(
                name="TransmittedSolar",
                key="OUTPUT:VARIABLE",
                var_name="Zone Windows Total Transmitted Solar Radiation Energy",
                freq="hourly",
                store_output=False,
            ),
        ]
        if timeseries_outputs != None:
            self.timeseries_outputs.extend(timeseries_outputs)
        # multiply len by 2 because perim/core
        self.sim_output_shape = (
            len([series for series in self.timeseries_outputs if series.store_output])
            * 2,
            8760,
        )

        self.storage_vec_len = 0
        self.ml_vec_len = 0
        self._key_ix_lookup = {}
        for i, parameter in enumerate(self.parameters):
            self._key_ix_lookup[parameter.name] = i
            parameter.start_storage = self.storage_vec_len
            self.storage_vec_len += parameter.len_storage
            if isinstance(parameter, SchedulesParameters) or not parameter.in_ml:
                parameter.start_ml = None
            else:
                parameter.start_ml = self.ml_vec_len
                self.ml_vec_len += parameter.len_ml

    @property
    def parameter_names(self):
        """Return a list of the named parameters in the schema"""
        return list(self._key_ix_lookup.keys())

    def __getitem__(self, key):
        """
        Args:
            key: str, name of parameter
        Returns:
            parameter: SchemaParameter

        """
        return self.parameters[self._key_ix_lookup[key]]

    def __str__(self):
        """Generate a summary of the storach schema"""
        desc = "-------- Schema --------"
        for parameter in self.parameters:
            desc += f"\n---- {parameter.name} ----"
            desc += f"\nshape storage: {parameter.shape_storage} / shape ml: {parameter.shape_ml}"
            desc += f"\nlocation storage: {parameter.start_storage}->{parameter.start_storage+parameter.len_storage} / location ml: {parameter.start_ml if parameter.in_ml and not isinstance(parameter, SchedulesParameters) else None}->{parameter.start_ml+parameter.len_ml if parameter.in_ml and not isinstance(parameter, SchedulesParameters) else None}"
            desc += f"\n"

        desc += f"\nTotal length of storage vectors: {self.storage_vec_len} / Total length of ml vectors: {self.ml_vec_len}"
        return desc

    def generate_empty_storage_vector(self):
        """
        Create a vector of zeros representing a blank storage vector

        Returns:
            storage_vector: np.ndarray, 1-dim, shape=(len(storage_vector))
        """
        empty_vec = np.zeros(shape=self.storage_vec_len)
        schedules = self["schedules"].extract_storage_values(empty_vec)
        schedules[:, SchedulesParameters.op_indices["scale"]] = 1
        self.update_storage_vector(empty_vec, "schedules", schedules)
        return empty_vec

    def generate_empty_storage_batch(self, n):
        """
        Create a matrix of zeros representing a batch of blank storage vectors

        Args:
            n: number of vectors to initialize in batch
        Returns:
            storage_batch: np.ndarray, 2-dim, shape=(n_vectors_in_batch, len(storage_vector))
        """
        # TODO: implement schedule ops initializer as fn in schedule.py instead of doing it here
        empty_tensor = np.zeros(shape=(n, self.storage_vec_len))
        schedules = self["schedules"].extract_storage_values_batch(empty_tensor)
        schedules[:, :, SchedulesParameters.op_indices["scale"]] = 1
        return empty_tensor

    def update_storage_vector(self, storage_vector, parameter, value):
        """
        Update a storage vector parameter with a value (or matrix which will be flattened)

        Args:
            storage_vector: np.ndarray, 1-dim, shape=(len(storage_vector))
            parameter: str, name of parameter to update
            value: np.ndarray | float, n-dim, will be flattened and stored in the storage vector
        """
        parameter = self[parameter]
        value = parameter.clip(value)
        start = parameter.start_storage
        end = start + parameter.len_storage
        if isinstance(value, np.ndarray):
            storage_vector[start:end] = value.flatten()
        else:
            storage_vector[start] = value

    def update_storage_batch(
        self, storage_batch, index=None, parameter=None, value=None
    ):
        """
        Update a storage vector parameter within a batch of storage vectors with a new value (or matrix which will be flattened)

        Args:
            storage_batch: np.ndarray, 2-dim, shape=(n_vectors, len(storage_vector))
            index: int | tuple, which storage vector (or range of storage vectors) within the batch to update.  omit or use None if updating the full batch
            parameter: str, name of parameter to update
            value: np.ndarray | float, n-dim, will be flattened and stored in the storage vector
        """
        parameter = self[parameter]
        value = parameter.clip(value)
        start = parameter.start_storage
        end = start + parameter.len_storage

        if isinstance(value, np.ndarray):
            value = value.reshape(-1, parameter.len_storage)

        if isinstance(index, tuple):
            start_ix = index[0]
            end_ix = index[1]
            storage_batch[start_ix:end_ix, start:end] = value
        else:
            if index == None:
                storage_batch[:, start:end] = value
            else:
                storage_batch[index, start:end] = value

    def to_ml(self, storage_batch):
        ml_vector_components = []
        timeseries_ops = None
        for parameter in self.parameters:
            if parameter.in_ml:
                vector_components = parameter.to_ml(storage_batch)
                if isinstance(parameter, SchedulesParameters):
                    timeseries_ops = vector_components
                else:
                    ml_vector_components.append(vector_components)
        ml_vectors = np.hstack(ml_vector_components)
        return ml_vectors, timeseries_ops

    def extract_from_template(self, building_template):
        # storage_vector = self.generate_empty_storage_vector()
        template_vect = []
        schedules_vect = []
        parameters = self.parameters
        for parameter in parameters:
            if parameter.in_ml:
                # print("Extracting value(s) for ", parameter.name)
                if isinstance(parameter, SchedulesParameters):
                    schedules_vect = parameter.extract_from_template(building_template)
                    # print(f"Got schedules of {type(schedules_vect)} with shape {schedules_vect.shape}")
                elif (
                    isinstance(parameter, BuildingTemplateParameter)
                    or isinstance(parameter, TMassParameter)
                    or isinstance(parameter, RValueParameter)
                ):  # TODO: check
                    val = parameter.extract_from_template(building_template)
                    # append to template vector
                    template_vect.append(val)
                elif isinstance(parameter, WindowParameter):
                    val = parameter.extract_from_template(building_template)
                    template_vect.extend(val)

        # return values which will be used for a building parameter vector and/or timeseries vector (schedules)
        return dict(
            template_vect=np.array(template_vect),
            schedules_vect=schedules_vect,
        )


if __name__ == "__main__":
    schema = Schema()
    print(schema)

    """Create a single empty storage vector"""
    storage_vector = schema.generate_empty_storage_vector()

    """Create a batch matrix of empty storage vectors"""
    batch_size = 20
    storage_batch = schema.generate_empty_storage_batch(batch_size)

    """
    Updating a storage batch with a constant parameter
    """
    schema.update_storage_batch(storage_batch, parameter="FacadeRValue", value=2)
    print(schema["FacadeRValue"].extract_storage_values_batch(storage_batch))

    """Updating a subset of a storage batch with random values"""
    start = 2
    n = 8
    end = start + n
    parameter = "PartitionRValue"
    shape = (n, *schema[parameter].shape_storage)
    values = np.random.rand(*shape)  # create a random sample with appropriate shape
    schema.update_storage_batch(
        storage_batch, index=(start, end), parameter=parameter, value=values
    )
    print(
        schema[parameter].extract_storage_values_batch(storage_batch)[
            start - 1 : end + 1
        ]
    )  # use [1:11] slice to see that the adjacentt cells are still zero

    """Updating an entire batch with random values"""
    parameter = "SlabRValue"
    n = batch_size
    shape = (n, *schema[parameter].shape_storage)
    values = np.random.rand(*shape)  # create a random sample with appropriate shape
    schema.update_storage_batch(storage_batch, parameter=parameter, value=values)
    print(schema[parameter].extract_storage_values_batch(storage_batch))
