import os
import sys
from glob import glob
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
import logging

module_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".")
)
# module_path = os.path.abspath(os.path.join("."))
# module_path = Path(module_path, "ml-for-bem")
if module_path not in sys.path:
    sys.path.append(str(module_path))

logging.basicConfig()
logger = logging.getLogger("WhiteBoxSim")
logger.setLevel(logging.INFO)

try:
    from archetypal import UmiTemplateLibrary
    from archetypal.idfclass.sql import Sql
    import pandas as pd

    # from pyumi.shoeboxer.shoebox import ShoeBox
    from pyumi.epw import EPW
except (ImportError, ModuleNotFoundError) as e:
    logger.error("Failed to import a package! Be wary about continuing...", exc_info=e)

from utils.constants import *
from utils.nrel_uitls import CLIMATEZONES_LIST, RESTYPES
from shoeboxer.shoebox_config import ShoeboxConfiguration
from shoeboxer.builder import ShoeBox

data_path = Path(os.path.dirname(os.path.abspath(__file__))) / "data"
SHOEBOX_PATH = Path(module_path, SHOEBOX_RELATIVE_PATH)

constructions_lib_path = os.path.join(
    os.getcwd(),
    "ml-for-bem",
    "data",
    "template_libs",
    "ConstructionsLibrary.json",
)


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
        "idf",
    )

    def __init__(self, schema, storage_vector, template_lib_path=None, template_idx=0):
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
        if template_lib_path:
            self.load_template(template_lib_path, template_idx)
        self.build_epw_path()
        self.update_parameters()
        self.build_shoebox()

    def load_template(self, template_lib_path, template_idx):
        """
        Method for loading a template based off id in storage vector.
        """

        self.lib = UmiTemplateLibrary.open(template_lib_path)
        self.template = self.lib.BuildingTemplates[template_idx]

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
        # TODO: switch to global epws
        globber = (
            data_path / "epws" / "city_epws_indexed" / f"cityidx_{int(cityidx):04d}**"
        )
        files = glob(str(globber))
        self.epw_path = data_path / files[0]

    def load_epw(self):
        self.epw = EPW(self.epw_path)

    def build_shoebox(self):
        sbid = int(
            self.schema["variation_id"].extract_storage_values(self.storage_vector)
        )
        sb = ShoeBox.from_vector(
            name=f"shoebox_{sbid}",
            schema=self.schema,
            shoebox_config=self.shoebox_config,
            vector=self.storage_vector,
            schedules=self.schema.parameters[-1].extract_from_template(
                self.template
            ),  # TODO fetch SchedulesParameter better
            change_summary=False,
            output_directory=SHOEBOX_PATH,
        )
        self.shoebox = sb

    def simulate(self):
        # self.shoebox.simulate(verbose=False, prep_outputs=False, readvars=False)
        idf = self.shoebox.idf(run_simulation=False)
        idf.simulate(verbose=False, prep_outputs=False, readvars=False)
        sql = Sql(idf.sql_file)
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
        series_to_retrieve_monthly.extend(series_to_retrieve_hourly)
        ep_df_monthly = pd.DataFrame(
            sql.timeseries_by_name(
                series_to_retrieve_monthly, reporting_frequency="Monthly"
            )
        )
        self.hourly = ep_df_hourly
        self.monthly = ep_df_monthly
        self.idf = idf  # TODO do we need this?
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
        print("Floor2Facade", self.shoebox_config.floor_2_facade)
        print("Core2Perim", self.shoebox_config.core_2_perim)
        print("Foot2Gnd [adia %]", self.shoebox_config.ground_2_footprint)
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
            "U Window:",
            self.template.Windows.Construction.u_value
            # "U Window:", self.template.Windows.Construction.Layers[0].Material.Uvalue
        )  # TODO: this is slightly different!) #TODO update this to read idf
        # print(
        #     "VLT",
        #     self.template.Windows.Construction.Layers[0].Material.VisibleTransmittance,
        # )
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
