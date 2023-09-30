"""Module for dynamically building shoebox idfs from a seed JSON file, shoebox-template.JSON."""

import json
import logging
from pathlib import Path
import subprocess
import os
import sys
import jsondiff
import copy
from enum import IntEnum
import math

module_path = os.path.abspath(os.path.join(".."))
module_path = Path(module_path, "ml-for-building-energy-modeling", "ml-for-bem")
if module_path not in sys.path:
    sys.path.append(str(module_path))

import archetypal as ar
from archetypal.idfclass import IDF
from archetypal import settings
from archetypal.idfclass.sql import Sql
from archetypal.template.schedule import UmiSchedule

from schedules import mutate_timeseries
from schema import TimeSeriesOutput

from shoebox_config import ShoeboxConfiguration
import geometry_utils as gu

import numpy as np
import pandas as pd

settings.energyplus_location = Path("D:\EnergyPlusV22-2-0")
settings.ep_version = "22.2.0"

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HRV(IntEnum):
    NoHRV = 0
    Sensible = 1
    Enthalpy = 2


class Econ(IntEnum):
    NoEconomizer = 0
    DifferentialDryBulb = 1
    DifferentialEnthalpy = 2


class MechVentModeSched(IntEnum):
    Off = 0
    AllOn = 1
    OccupancySchedule = 2


class BooleanParam(IntEnum):
    false = 0
    true = 1


TIMESERIES_OUTPUTS = [
    TimeSeriesOutput(
        name="Heating",
        key="OUTPUT:VARIABLE",
        var_name="Zone Ideal Loads Supply Air Total Heating Energy",
        freq="Hourly",
        store_output=True,
    ),
    TimeSeriesOutput(
        name="Cooling",
        key="OUTPUT:VARIABLE",
        var_name="Zone Ideal Loads Supply Air Total Cooling Energy",
        freq="Hourly",
        store_output=True,
    ),
    TimeSeriesOutput(
        name="Lighting",
        key="OUTPUT:VARIABLE",
        var_name="Lights Total Heating Energy",
        freq="Hourly",
        store_output=False,
    ),
    TimeSeriesOutput(
        name="TransmittedSolar",
        key="OUTPUT:VARIABLE",
        var_name="Zone Windows Total Transmitted Solar Radiation Energy",
        freq="Hourly",
        store_output=False,
    ),
]

SCHEDULE_PATHS = [
    ["Loads", "EquipmentAvailabilitySchedule"],
    ["Loads", "LightsAvailabilitySchedule"],
    ["Loads", "OccupancySchedule"],
    # ["Conditioning", "MechVentSchedule"],
    # ["Conditioning", "CoolingSchedule"],
    # ["Conditioning", "HeatingSchedule"],
    # ["Conditioning", "HeatingSetpointSchedule"],
    # ["Conditioning", "HeatingSetpointSchedule"],
    # ["DomesticHotWater", "WaterSchedule"],
    # ["Ventilation", "NatVentSchedule"],
    # ["Ventilation", "ScheduledVentilationSchedule"],
    # ["Windows", "ZoneMixingAvailabilitySchedule"],
    # ["Windows", "ShadingSystemAvailabilitySchedule"],
    # ["Windows", "AfnWindowAvailabilitySchedule"],
]


def template_dict(
    schedules,
    people_density=0.05,
    lighting_power_density=2,
    equipment_power_density=5,
    infiltration_per_area=0.0001,
    ventilation_per_floor_area=0.0004,
    ventilation_per_person=0.025,
    ventilation_mode=0,  # one hot of 0-2
    heating_sp=18,
    cooling_sp=24,
    humid_max=81,
    humid_min=21,
    sat_max=28,
    sat_min=17,
    heat_recovery=0,
    economizer=1,
    wall_r_val=2,
    wall_mass=8000,
    roof_r_val=6,
    roof_mass=200000,
    slab_r_val=4,
    shgc=0.5,
    window_u_val=1.0,
    visible_transmittance=0.8,
):
    return dict(
        schedules=schedules,
        people_density=people_density,
        lighting_power_density=lighting_power_density,
        equipment_power_density=equipment_power_density,
        infiltration_per_area=infiltration_per_area,
        ventilation_per_floor_area=ventilation_per_floor_area,
        ventilation_per_person = ventilation_per_person,
        ventilation_mode=ventilation_mode,
        heating_sp=heating_sp,
        cooling_sp=cooling_sp,
        humid_max=humid_max,
        humid_min=humid_min,
        sat_max=sat_max,
        sat_min=sat_min,
        heat_recovery=heat_recovery,
        economizer=economizer,
        wall_r_val=wall_r_val,
        wall_mass=wall_mass,
        roof_r_val=roof_r_val,
        roof_mass=roof_mass,
        slab_r_val=slab_r_val,
        shgc=shgc,
        window_u_val=window_u_val,
        visible_transmittance=visible_transmittance,
    )


def get_template_dict_from_archetypal(template):
    pass #TODO


class ShoeBox:
    def __init__(
        self,
        name,
        shoebox_config: ShoeboxConfiguration,
        epw,
        seed=Path(module_path, "shoeboxer/shoebox-template.json"),
        output_directory=None,
    ):
        self.name = name
        self.output_directory = output_directory
        self.shoebox_config = shoebox_config
        self.epw = epw
        self.hourly = None
        self.monthly = None

        perim_depth = shoebox_config.floor_2_facade * shoebox_config.height
        core_depth = shoebox_config.core_2_perim * perim_depth
        length = perim_depth + core_depth
        self.floor_area = shoebox_config.width * length

        # Load the seed model
        with open(seed, "r") as f:
            self._seed_epjson = json.load(f)
            self.epjson = copy.deepcopy(self._seed_epjson)

        # Make a copy
        if self.output_directory:
            fname = self.output_directory / f"{self.name}.epjson"
        else:
            fname = f"{self.name}.epjson"

        self.ep_json_path = fname

        with open(self.ep_json_path, "w") as f:
            json.dump(self.epjson, f, indent=4)

    def update_epjson(self, template_dict, change_summary=True):
        # Rotate North
        new_north_deg = self.shoebox_config.orientation * 90
        self.rotate_relative_north(new_north_deg)
        # Update characteristics based on archetypal template
        self.handle_template(template_dict)
        # Save new epJSON in output directory or cache
        with open(self.ep_json_path, "w") as f:
            json.dump(self.epjson, f, indent=4)
        # Turn into an IDF for geometry handling
        idf = self.idf(run_simulation=False)
        # Update geometry
        # self.handle_geometry()
        idf = self.handle_shading(idf)
        if change_summary:

            # TODO: get idf as a json again
            self.compare_idfs()
        return idf

    def rotate_relative_north(self, orient):
        self.epjson["Building"]["Building"]["north_axis"] = int(orient)
        logger.info(
            f"Changed orientation of relative north to {self.epjson['Building']['Building']['north_axis']}"
        )

    def handle_template(self, template_dict):
        """
        Takes in dictionary of template values
        """
        self.handle_constructions(template_dict)
        self.handle_windows(template_dict)
        self.handle_equipment(template_dict)
        self.handle_lights(template_dict)
        self.handle_people(template_dict)
        self.handle_infiltration(template_dict)
        self.handle_hvac_econ_enthalpy(template_dict)
        # self.handle_sat_minmax(template_dict)
        # self.handle_humidistat(template_dict) # TODO: are we using?
        self.handle_ventilation(template_dict)
        self.handle_thermostat(template_dict)

    def handle_constructions(self, template_dict):
        """
        Update Roof and Wall constructions
        """
        # TODO: Fetch based off of building surfaces with outdoor boundary conditions?
        for con_name, con_def in self.epjson["Construction"].items():
            if "Exterior Wall" in con_name:
                self.change_construction_r(con_def, template_dict["wall_r_val"])
            elif "Exterior Roof" in con_name:
                self.change_construction_r(con_def, template_dict["roof_r_val"])
            elif "Ground Slab" in con_name:
                self.change_construction_r(con_def, template_dict["roof_r_val"])
            # TODO: change construction mass

    def change_construction_r(self, construction, new_r):
        """
        Change a Construction's insulation layer to reach a target u
        """
        r_vals = []
        for layer_name in construction.values():
            material_def = self.epjson["Material"][layer_name]
            k = material_def["conductivity"]
            thick = material_def["thickness"]
            r_val = 1 / (k / thick)
            r_vals.append((layer_name, r_val, material_def))

        # TODO: U-Vals come out the same, but R is slightly different?
        current_r_val = sum(r_val for _, r_val, _ in r_vals)
        sorted_layers = sorted(r_vals, key=lambda x: -x[1])
        insulator = sorted_layers[0]
        r_insulator_current = insulator[1]
        insulator_def = insulator[2]
        r_val_without_insulator = current_r_val - r_insulator_current
        needed_r = new_r - r_val_without_insulator
        assert needed_r > 0
        new_thickness = needed_r * insulator_def["conductivity"]
        insulator_def["thickness"] = round(new_thickness, 3)

    def handle_windows(self, template_dict):
        """
        Update windows in epjson
        """

        # Create the Material
        window_material_type = "WindowMaterial:SimpleGlazingSystem"
        window_material_def = {
            "solar_heat_gain_coefficient": template_dict["shgc"],
            "u_factor": template_dict["window_u_val"],
            "visible_transmittance": template_dict["visible_transmittance"],
        }
        window_material_name = f"SimpleGlazing"
        if window_material_type not in self.epjson:
            self.epjson[window_material_type] = {}
        self.epjson[window_material_type][window_material_name] = window_material_def

        # Create the construction
        window_construction_name = f"defaultGlazing"
        window_construction_def = {"outside_layer": window_material_name}
        self.epjson["Construction"][window_construction_name] = window_construction_def

        # Update all the window's constructions
        for fen_srf_def in self.epjson["FenestrationSurface:Detailed"].values():
            fen_srf_def["construction_name"] = window_construction_name

    def handle_lights(self, template_dict):
        lights_var_def = self.epjson["Lights"]["SharedLights"]
        lights_var_def["watts_per_zone_floor_area"] = template_dict[
            "lighting_power_density"
        ]
        # TODO: schedules

    def handle_people(self, template_dict):
        people_def = self.epjson["People"]["SharedPeople"]
        people_def["people_per_floor_area"] = template_dict["people_density"]
        # TODO: schedules

    def handle_equipment(self, template_dict):
        """
        Update Equipment
        """

        # equipment_base_def = self.epjson["ElectricEquipment"][
        #     "SharedElectricEquipmentBaseload"
        # ]
        # equipment_base_def["watts_per_zone_floor_area"] = template_dict["equipment_power_density"]
        # equipment_base_def["schedule_name"] = "AllOn"

        equipment_var_def = self.epjson["ElectricEquipment"]["SharedElectricEquipment"]
        equipment_var_def["watts_per_zone_floor_area"] = template_dict[
            "equipment_power_density"
        ]

        idx = [SCHEDULE_PATHS.index(i) for i in SCHEDULE_PATHS if "Equipment" in i[1]][
            0
        ]
        schedule_array = template_dict["schedules"][idx, :]
        year, week, day = UmiSchedule.from_values(
            Name="EquipmentSchedule", Values=schedule_array
        ).to_year_week_day()
        logger.info(year)
        # logger.info(year.to_dict())
        # for i, sched_path in enumerate(SCHEDULE_PATHS):
        #     schedule_array = template_dict['schedules'][i, :]
        # if "EQUIPMENT" in sched_path[1].upper():

    def handle_schedules(self):
        """
        Update all schedules in the idf with archetypal
        """
        pass

    def handle_infiltration(self, template_dict):
        """
        Update infiltration specification
        """
        for infil_def in self.epjson["ZoneInfiltration:DesignFlowRate"].values():
            infil_def["design_flow_rate_calculation_method"] = "Flow/ExteriorArea"
            infil_def["flow_rate_per_exterior_surface_area"] = template_dict[
                "infiltration_per_area"
            ]

    def handle_hvac_econ_enthalpy(self, template_dict):
        """
        Update Basic HVAC stuff
        # TODO: Should we also be updating the Sizing:Zone object?
        """
        for zone in self.epjson["ZoneHVAC:IdealLoadsAirSystem"].values():
            # Handle Heat Recovery
            if template_dict["heat_recovery"] == HRV.NoHRV:
                zone["heat_recovery_type"] = "None"
            elif template_dict["heat_recovery"] == HRV.Sensible:
                zone["sensible_heat_recovery_effectiveness"] = 0.7
                zone["heat_recovery_type"] = "Sensible"
            elif template_dict["heat_recovery"] == HRV.Enthalpy:
                zone["sensible_heat_recovery_effectiveness"] = 0.7
                zone["latent_heat_recovery_effectiveness"] = 0.65
                zone["heat_recovery_type"] = "Enthalpy"

            # Handle Economizer
            zone["outdoor_air_economizer_type"] = Econ(template_dict['economizer']).name

    def handle_sat_minmax(self, template_dict):
        """
        Handle supply air temperature min and max
        """
        for zone in self.epjson["ZoneHVAC:IdealLoadsAirSystem"].values():
            zone["maximum_heating_supply_air_temperature"] = template_dict["sat_max"]
            zone["minimum_cooling_supply_air_temperature"] = template_dict["sat_min"]

    def handle_humidistat(self, template_dict):
        """
        Update Humidity Min/Max Controls
        """
        # Handle Humidistat Controls
        for zone in self.epjson["ZoneHVAC:IdealLoadsAirSystem"].values():
            # Make sure controls are set
            zone["dehumidification_control_type"] = "Humidistat"
            zone["humidification_control_type"] = "Humidistat"

        # Figure out which schedules need to be changed
        humidimax_scheds_to_change = set()
        humidimin_scheds_to_change = set()
        for humidistat in self.epjson["ZoneControl:Humidistat"].values():
            humidimax_scheds_to_change.add(
                humidistat["dehumidifying_relative_humidity_setpoint_schedule_name"]
            )
            humidimin_scheds_to_change.add(
                humidistat["humidifying_relative_humidity_setpoint_schedule_name"]
            )

        # Find those schedules and update them
        for schedule_name in humidimax_scheds_to_change:
            self.epjson["Schedule:Constant"][schedule_name]["hourly_value"] = template_dict["humid_max"]

        for schedule_name in humidimin_scheds_to_change:
            self.epjson["Schedule:Constant"][schedule_name]["hourly_value"] = template_dict["humid_min"]

    def handle_ventilation(self, template_dict):
        """
        Update Ventilation

        If DCV is off, then the ventilation rate is set to the following:
        (floor_area * flow_per_area + flow_per_person * design_occ) * OA_schedule

        If DCV is on, then the ventilation rate is set to the following:
        (floor_area * flow_per_area + flow_per_person * design_occ * Occ_Schedule) * OA_schedule

        """

        mech_vent_sched_name = MechVentModeSched(template_dict['ventilation_mode']).name
        
        logger.info(f"Mechanical ventilation schedule: {mech_vent_sched_name}")
        # TODO: check outputs of when ventilation is on
        self.epjson["DesignSpecification:OutdoorAir"]["SharedDesignSpecOutdoorAir"] = {
            "outdoor_air_flow_per_person": template_dict["ventilation_per_person"],
            "outdoor_air_flow_per_zone_floor_area": template_dict["ventilation_per_floor_area"],
            "outdoor_air_method": "Sum",
            "outdoor_air_schedule_name": "", # TODO will this ever change?
        }

        if mech_vent_sched_name == MechVentModeSched.OccupancySchedule:
            for zone in self.epjson["ZoneHVAC:IdealLoadsAirSystem"].values():
                zone["demand_controlled_ventilation_type"] = mech_vent_sched_name
        else:
            for zone in self.epjson["ZoneHVAC:IdealLoadsAirSystem"].values():
                zone["demand_controlled_ventilation_type"] = "None"

    def handle_thermostat(self, template_dict):
        # TODO: allow for schedule?
        self.epjson["Schedule:Constant"]["CoolingSPSchedule"]["hourly_value"] = template_dict["cooling_sp"]
        self.epjson["Schedule:Constant"]["HeatingSPSchedule"]["hourly_value"] = template_dict["heating_sp"]

    def handle_geometry(self):
        # scale geometry
        gu.set_adiabatic_surfaces()
        # Update window to wall ratio
        gu.update_wwr(shoebox_config.wwr)

    def handle_shading(self, idf):
        r = 2 * self.shoebox_config.width
        idf = gu.build_shading(idf, angles=self.shoebox_config.shading_vect, radius=r, override=False)
        return idf
    
    def idf(self, run_simulation=True):
        idf_path = self.convert()
        idf = IDF(idf_path, epw=self.epw)
        if run_simulation:
            self.hourly, self.monthly = self.simulate(idf)
        return idf

    @classmethod
    def simulate(cls, idf):
        idf.simulate(verbose=False, prep_outputs=False, readvars=False)
        sql = Sql(idf.sql_file)
        series_to_retrieve_hourly = []
        series_to_retrieve_monthly = []
        for timeseries in TIMESERIES_OUTPUTS:
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
        return ep_df_hourly, ep_df_monthly

    def convert(self):
        # Define the command and its arguments
        cmd = settings.energyplus_location / "energyplus.exe"
        args = ["--convert-only", self.ep_json_path]

        # TODO change location of idf

        # Run the command
        with subprocess.Popen(
            [cmd] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ) as proc:
            for line in proc.stdout:
                logger.info(line.strip())
            exit_code = proc.wait()

        # Check if the command was successful
        if exit_code == 0:
            logger.info("Command executed successfully.")
        else:
            logger.error(f"Command failed with exit code {exit_code}.")
            raise RuntimeError(f"Failed to convert EpJSON to IDF.")

        # return str(fname).replace("epjson", "idf")
        return f"{self.name}.idf"

    def compare_idfs(self):
        diff_report = jsondiff.diff(self.epjson, self._seed_epjson, syntax="symmetric")
        if diff_report:
            logger.info(json.dumps(diff_report, indent=2))
        else:
            logger.info("No changes in JSON found!")


if __name__ == "__main__":
    shoebox_config = ShoeboxConfiguration()
    shoebox_config.width = 3.0
    shoebox_config.height = 4.5
    shoebox_config.floor_2_facade = 0.5
    shoebox_config.core_2_perim = 0.5
    shoebox_config.roof_2_footprint = 0.5
    shoebox_config.ground_2_footprint = 0.5
    shoebox_config.wwr = 0.4
    shoebox_config.orientation = 2
    shoebox_config.template_name = "baseline"
    shoebox_config.shading_vect = np.random.random(12) * math.pi / 2

    # MAKE FAKE SCHEDULES
    schedules = np.zeros((3, 21))
    # UNIRAND
    # Equipment
    schedules[0, 18] = 1  # hours per sample
    schedules[0, 17] = 24 * 7  # samples per pattern
    schedules[0, 16] = 0  # continuous
    # Occupancy
    schedules[1, 18] = 1  # hours per sample
    schedules[1, 17] = 24  # samples per pattern
    schedules[1, 16] = 0  # continuous
    # Lights
    schedules[2, 18] = 1  # hours per sample
    schedules[2, 17] = 22  # samples per pattern
    schedules[2, 16] = 0  # continuous
    # build timeseries
    scheds = mutate_timeseries(np.ones((3, 8760)), schedules, 0)

    epw = "D:/Users/zoelh/GitRepos/ml-for-building-energy-modeling/ml-for-bem/data/epws/city_epws_indexed/cityidx_0001_USA_NY-New York Central Prk Obs Belv.725033_TMY3.epw"
    out_dir = Path("./ml-for-bem/shoeboxer/cache")

    sb = ShoeBox(
        name="test", shoebox_config=shoebox_config, epw=epw, output_directory=out_dir
    )
    idf = sb.update_epjson(template_dict(scheds))

    # idf = sb.idf(run_simulation=False)
    # print("HEATING/COOLING EUI")
    # print(sb.monthly.sum()/sb.floor_area*2.77e-07)
    idf.view_model()