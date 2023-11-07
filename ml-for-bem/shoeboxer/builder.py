"""Module for dynamically building shoebox idfs from a seed JSON file, shoebox-template.JSON."""

import calendar
import copy
import json
import logging
import math
import os
import subprocess
from pathlib import Path

import archetypal as ar
import jsondiff
import numpy as np
import pandas as pd
from archetypal import settings
from archetypal.idfclass import IDF
from archetypal.idfclass.sql import Sql
from archetypal.schedule import Schedule

import shoeboxer.geometry_utils as gu
from shoeboxer.shoebox_config import ShoeboxConfiguration
from utils.constants import *
from utils.schedules import mutate_timeseries

# from schema import Schema


EPW_PATH = Path(EPW_RELATIVE_PATH)

logging.basicConfig()
logger = logging.getLogger("ShoeBox")
logger.setLevel(logging.INFO)


def day_to_epbunch(dsched, idx=0, sched_lim=sched_type_limits):
    return {
        dsched.Name: dict(
            **{"hour_{}".format(i + 1): dsched.all_values[i] for i in range(24)},
            schedule_type_limits_name=sched_lim["Name"],
        )
    }


def week_to_epbunch(wsched, idx=0, sched_lim=sched_type_limits):
    return {
        wsched.Name: dict(
            **{
                f"{calendar.day_name[i].lower()}_schedule_day_name": day.Name
                for i, day in enumerate(wsched.Days)
            },
            holiday_schedule_day_name=wsched.Days[6].Name,
            summerdesignday_schedule_day_name=wsched.Days[0].Name,
            winterdesignday_schedule_day_name=wsched.Days[0].Name,
            customday1_schedule_day_name=wsched.Days[1].Name,
            customday2_schedule_day_name=wsched.Days[6].Name,
        )
    }


def year_to_epbunch(sched, sched_lim=sched_type_limits):
    dict_list = []
    for i, part in enumerate(sched.Parts):
        dict_list.append(
            dict(
                **{
                    "schedule_week_name": part.Schedule.Name,
                    "start_month": part.FromMonth,
                    "start_day".format(i + 1): part.FromDay,
                    "end_month".format(i + 1): part.ToMonth,
                    "end_day".format(i + 1): part.ToDay,
                }
            )
        )
    return dict(
        schedule_type_limits_name=sched_lim["Name"],
        schedule_weeks=dict_list,
    )


def schedule_to_epbunch(name, values, sched_lims_bunch=sched_type_limits):
    assert len(values) == 8760, "Schedule length does not equal 8760 hours!"
    arch_schedule = Schedule(Name=name, Values=values)
    y, w, d = arch_schedule.to_year_week_day()
    year_bunch = year_to_epbunch(y, sched_lims_bunch)
    week_bunches = []
    day_bunches = []
    for week in w:
        week_bunches.append(week_to_epbunch(week, sched_lims_bunch))
    for day in d:
        day_bunches.append(day_to_epbunch(day, sched_lims_bunch))
    return year_bunch, week_bunches, day_bunches


def template_dict(
    schedules,
    PeopleDensity: float,
    LightingPowerDensity: float,
    EquipmentPowerDensity: float,
    Infiltration: float,  # m3/s/m2 of exterior exposure
    VentilationPerArea: float,
    VentilationPerPerson: float,
    VentilationMode: int,  # one hot of 0-2
    HeatingSetpoint: float,
    CoolingSetpoint: float,
    # humid_max: float,
    # humid_min: float,
    # sat_max: float,
    # sat_min: float,
    RecoverySettings: int,
    EconomizerSettings: int,
    FacadeRValue: float,
    FacadeMass: int,  # need to revisit minimums
    RoofRValue: float,
    RoofMass: int,
    SlabRValue: float,
    WindowShgc: float,
    WindowUValue: float,
    # visible_transmittance=0.8,
):
    return dict(
        schedules=schedules,
        PeopleDensity=PeopleDensity,
        LightingPowerDensity=LightingPowerDensity,
        EquipmentPowerDensity=EquipmentPowerDensity,
        Infiltration=Infiltration,
        VentilationPerArea=VentilationPerArea,
        VentilationPerPerson=VentilationPerPerson,
        VentilationMode=VentilationMode,
        HeatingSetpoint=HeatingSetpoint,
        CoolingSetpoint=CoolingSetpoint,
        # humid_max=humid_max,
        # humid_min=humid_min,
        # sat_max=sat_max,
        # sat_min=sat_min,
        RecoverySettings=RecoverySettings,
        EconomizerSettings=EconomizerSettings,
        FacadeRValue=FacadeRValue,
        FacadeMass=FacadeMass,
        RoofRValue=RoofRValue,
        RoofMass=RoofMass,
        SlabRValue=SlabRValue,
        WindowShgc=WindowShgc,
        WindowUValue=WindowUValue,
        # visible_transmittance=visible_transmittance,
    )


class ShoeBox:
    def __init__(
        self,
        name,
        shoebox_config: ShoeboxConfiguration,
        epw,
        template_dict,
        seed_model=Path("shoeboxer/shoebox-template.json"),
        output_directory=None,
        change_summary=False,
        verbose=False,
    ):
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        self.name = name
        # Check if output directory exists and create if not
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        self.output_directory = output_directory
        self.shoebox_config = shoebox_config  # TODO if this changes, update the json
        self.epw = epw
        # self.floor_2_facade = shoebox_config.perim_depth / shoebox_config.height
        # self.core_2_perim = shoebox_config.core_depth / shoebox_config.perim_depth
        # perim_depth = shoebox_config.floor_2_facade * shoebox_config.height
        # core_depth = shoebox_config.core_2_perim * shoebox_config.perim_depth
        total_depth = shoebox_config.perim_depth + shoebox_config.core_depth
        self.floor_area = shoebox_config.width * total_depth

        # Load the seed model
        with open(seed_model, "r") as f:
            self._seed_epjson = json.load(f)
            self.epjson = copy.deepcopy(self._seed_epjson)

        # Make a copy
        if self.output_directory:
            fname = self.output_directory / f"{self.name}.epjson"
        else:
            fname = f"{self.name}.epjson"
        self.ep_json_path = fname
        self.save_json()
        self.update_epjson(template_dict, change_summary=change_summary)

    @classmethod
    def from_archetypal(cls, template):
        pass  # TODO make class function

    @classmethod
    def from_vector(cls, name, schema, shoebox_config, vector, schedules, **kwargs):
        # from schema import Schema - assert schema is schema
        epw_map = pd.read_csv(EPW_MAP_PATH, index_col=0)
        epw_idx = int(schema["base_epw"].extract_storage_values(vector))
        logger.info(f"Simulating in {epw_map.loc[epw_idx, 'city']}")
        epw = epw_map.loc[epw_idx, "slug"]

        # seed = int(schema["shading_seed"].extract_storage_values(vector))
        # np.random.seed(seed)
        # if hasattr(shoebox_config, "shading_vect") == False:
        #     logger.debug("Creating shading from seed.")
        #     shoebox_config.shading_vect = (
        #         np.random.random(SHADING_DIV_SIZE) * math.pi / 2.5
        #     )  # TODO how to divide this? Do this in schema
        td = template_dict(
            schedules,
            PeopleDensity=schema["PeopleDensity"].extract_storage_values(vector),
            LightingPowerDensity=schema["LightingPowerDensity"].extract_storage_values(
                vector
            ),
            EquipmentPowerDensity=schema[
                "EquipmentPowerDensity"
            ].extract_storage_values(vector),
            Infiltration=schema["Infiltration"].extract_storage_values(vector),
            VentilationPerArea=schema["VentilationPerArea"].extract_storage_values(
                vector
            ),
            VentilationPerPerson=schema["VentilationPerPerson"].extract_storage_values(
                vector
            ),
            VentilationMode=schema["VentilationMode"].extract_storage_values(vector),
            HeatingSetpoint=schema["HeatingSetpoint"].extract_storage_values(vector),
            CoolingSetpoint=schema["CoolingSetpoint"].extract_storage_values(vector),
            # humid_max=81,
            # humid_min=21,
            # sat_max=28,
            # sat_min=17,
            RecoverySettings=schema["RecoverySettings"].extract_storage_values(vector),
            EconomizerSettings=schema["EconomizerSettings"].extract_storage_values(
                vector
            ),
            FacadeRValue=schema["FacadeRValue"].extract_storage_values(vector),
            FacadeMass=schema["FacadeMass"].extract_storage_values(vector),
            RoofRValue=schema["RoofRValue"].extract_storage_values(vector),
            RoofMass=schema["RoofMass"].extract_storage_values(vector),
            SlabRValue=schema["SlabRValue"].extract_storage_values(vector),
            WindowShgc=schema["WindowShgc"].extract_storage_values(vector),
            WindowUValue=schema["WindowUValue"].extract_storage_values(vector),
            # visible_transmittance=0.8,  # TODO?
        )
        return cls(
            name=name,
            shoebox_config=shoebox_config,
            epw=Path(EPW_PATH, epw),
            template_dict=td,
            **kwargs,
        )

    def save_json(self):
        with open(self.ep_json_path, "w") as f:
            json.dump(self.epjson, f, indent=4)

    def update_epjson(self, template_dict, change_summary=True):
        # Rotate North
        # new_north_deg =  self.shoebox_config.orientation * 90
        self.rotate_relative_north(self.shoebox_config.orientation)

        # Update characteristics based on archetypal template
        self.handle_template(template_dict)

        # Save new epJSON in output directory or cache
        with open(self.ep_json_path, "w") as f:
            json.dump(self.epjson, f, indent=4)

        # Update geometry
        self.handle_geometry()
        self.handle_shading()

        # Turn into an IDF
        self.save_json()
        # idf.save()

        if change_summary:
            self.compare_idfs(self.epjson)
        #     # get idf as a json again
        #     json_path = self.convert(
        #         path=str(self.ep_json_path).replace("epjson", "idf"), file_type="epjson"
        #     )
        #     with open(json_path, "r") as f:
        #         self.compare_idfs(json.load(f))

        return self.epjson

    def rotate_relative_north(self, orient):
        self.epjson["Building"]["Building"]["north_axis"] = int(orient)
        logger.debug(
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
        # self.handle_humidistat(template_dict)
        self.handle_ventilation(template_dict)
        self.handle_thermostat(template_dict)

    def handle_constructions(self, template_dict):
        """
        Update Roof and Wall constructions
        """
        # TODO: Fetch based off of building surfaces with outdoor boundary conditions?
        for con_name, con_def in self.epjson["Construction"].items():
            if "Exterior Wall" in con_name:
                self.change_construction_r(con_def, template_dict["FacadeRValue"])
                self.change_construction_mass(
                    con_def, template_dict["FacadeMass"], template_dict["FacadeRValue"]
                )
            elif "Exterior Roof" in con_name:
                self.change_construction_r(con_def, template_dict["RoofRValue"])
                self.change_construction_mass(
                    con_def, template_dict["RoofMass"], template_dict["RoofRValue"]
                )
            elif "Ground Slab" in con_name:
                self.change_construction_r(con_def, template_dict["SlabRValue"])

    def calculate_r(self, construction):
        r_vals = []
        for layer_name in construction.values():
            material_def = self.epjson["Material"][layer_name]
            k = material_def["conductivity"]
            thick = material_def["thickness"]
            r_val = 1 / (k / thick)
            r_vals.append((layer_name, r_val, material_def))
        return r_vals

    def change_construction_r(self, construction, new_r):
        """
        Change a Construction's insulation layer to reach a target u
        """
        r_vals = self.calculate_r(construction)
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
        if new_thickness < 0:
            raise ValueError("Desired R-value and TM combo is not possible.")
        if new_thickness < 0.003:
            logger.warning(
                "Thickness of insulation is less than 0.003. This will create a warning in EnergyPlus."
            )
        insulator_def["thickness"] = round(new_thickness, 3)
        new_r_vals = self.calculate_r(construction)
        logger.debug(
            f"New R-val = {sum(r for _, r, _ in new_r_vals)} compared to desired {new_r}"
        )

    def calculate_tm(self, construction):
        tm_vals = []
        for layer_name in construction.values():
            material_def = self.epjson["Material"][layer_name]
            c = material_def["specific_heat"]
            thick = material_def["thickness"]
            rho = material_def["density"]
            tm_val = c * rho * thick
            tm_vals.append((layer_name, tm_val, material_def))
        return tm_vals

    def change_construction_mass(self, construction, tm_idx, new_r_val):
        """
        specific heat (c) of the material layer from template is in units of J/(kg-K)
        density is in kg/m3
        thermal mass is in J/Km2 TM = c * density * thickness
        """
        new_thermal_mass_type = ThermalMassConstructions(tm_idx).name
        if "ExteriorWall" in list(construction.values())[0]:
            construction["layer_2"] = f"ExteriorWall{new_thermal_mass_type}"
        elif "ExteriorRoof" in list(construction.values())[0]:
            construction["layer_3"] = f"ExteriorRoof{new_thermal_mass_type}"
        # Check if mass/rvalue combo is possible
        logger.debug("Recalculating r-values...")
        self.change_construction_r(construction, new_r_val)

        # new_thermal_mass = ThermalMassCapacities[new_thermal_mass_type].value
        # def calc_needed_tm(construction):
        #     tm_vals = self.calculate_tm(construction)
        #     current_tm_val = sum(tm for _, tm, _ in tm_vals)
        #     sorted_layers = sorted(tm_vals, key=lambda x: -x[1])
        #     mass_layer = sorted_layers[0]
        #     mass_layer_tm = mass_layer[1]
        #     mass_layer_def = mass_layer[2]
        #     mass_without_concrete = current_tm_val - mass_layer_tm
        #     needed_tm = new_thermal_mass - mass_without_concrete
        #     logger.debug(f"Current mass: {current_tm_val}")
        #     logger.debug(f"Needed mass: {new_thermal_mass}")
        #     logger.debug(f"Needed mass of concrete: {needed_tm}")
        #     logger.debug(f"mass_without_concrete: {mass_without_concrete}")
        #     return mass_layer, mass_layer_def, needed_tm

        # mass_layer, mass_layer_def, needed_tm = calc_needed_tm(construction)

        # # Check if mass/rvalue combo is possible
        # if needed_tm < 0 and "ExteriorWall" in list(construction.values())[0]:
        #     logger.info(
        #         "Light mass wall condition, replacing high-mass stucco with wood siding."
        #     )
        #     construction["outside_layer"] = "ExteriorWallWoodFrame"
        #     del construction["layer_4"]
        #     logger.debug("Recalculating r-values...")
        #     self.change_construction_r(construction, new_r_val)
        #     mass_layer, mass_layer_def, needed_tm = calc_needed_tm(construction)

        # new_thickness = (
        #     needed_tm / mass_layer_def["specific_heat"] / mass_layer_def["density"]
        # )
        # new_thickness = round(new_thickness, 3)
        # assert new_thickness > 0, "Desired mass is not possible with given r-value"
        # if new_thickness < 0.003:
        #     logger.warning(
        #         f"Thickness of insulation is less than 0.003, at {new_thickness}. This will raise a warning in EnergyPlus."
        #     )
        # logger.debug(f"New thickness of {new_thickness} for {mass_layer[0]}")
        # mass_layer_def["thickness"] = new_thickness
        # new_tm_vals = self.calculate_tm(construction)
        # logger.debug(
        #     f"New thermal mass = {sum(tm for _, tm, _ in new_tm_vals)} compared to desired {new_thermal_mass}"
        # )

    def handle_windows(self, template_dict):
        """
        Update windows in epjson
        """

        # Create the Material
        window_material_type = "WindowMaterial:SimpleGlazingSystem"
        window_material_def = {
            "solar_heat_gain_coefficient": template_dict["WindowShgc"],
            "u_factor": template_dict["WindowUValue"],
            # "visible_transmittance": template_dict["visible_transmittance"],
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
            "LightingPowerDensity"
        ]
        # Schedules
        lights_sched_name = lights_var_def["schedule_name"]
        assert lights_sched_name == "LightsSchedule"

        idx = [SCHEDULE_PATHS.index(i) for i in SCHEDULE_PATHS if "Lights" in i[1]][0]
        values = template_dict["schedules"][idx, :]
        self.handle_schedules(lights_sched_name, values)

    def handle_people(self, template_dict):
        people_def = self.epjson["People"]["SharedPeople"]
        people_def["people_per_floor_area"] = template_dict["PeopleDensity"]

        # Schedules
        people_sched_name = people_def["number_of_people_schedule_name"]
        assert people_sched_name == "OccupancySchedule"

        idx = [SCHEDULE_PATHS.index(i) for i in SCHEDULE_PATHS if "Occupancy" in i[1]][
            0
        ]
        values = template_dict["schedules"][idx, :]
        self.handle_schedules(people_sched_name, values)

    def handle_equipment(self, template_dict):
        """
        Update Equipment
        """

        # equipment_base_def = self.epjson["ElectricEquipment"][
        #     "SharedElectricEquipmentBaseload"
        # ]
        # equipment_base_def["watts_per_zone_floor_area"] = template_dict["EquipmentPowerDensity"]
        # equipment_base_def["schedule_name"] = "AllOn"

        equipment_var_def = self.epjson["ElectricEquipment"]["SharedElectricEquipment"]
        equipment_var_def["watts_per_zone_floor_area"] = template_dict[
            "EquipmentPowerDensity"
        ]

        equipment_var_sched_name = equipment_var_def["schedule_name"]
        assert equipment_var_sched_name == "EquipmentSchedule"

        idx = [SCHEDULE_PATHS.index(i) for i in SCHEDULE_PATHS if "Equipment" in i[1]][
            0
        ]
        values = template_dict["schedules"][idx, :]
        self.handle_schedules(equipment_var_sched_name, values)

    def handle_schedules(self, sched_name, values):
        """
        Update all schedules in the idf with archetypal
        """
        year_bunch, week_bunches, day_bunches = schedule_to_epbunch(
            sched_name, values, sched_lims_bunch=sched_type_limits
        )
        # Remove all schedules containing base name
        # previous_day_names = []
        previous_week_names = [
            x["schedule_week_name"]
            for x in self.epjson["Schedule:Year"][sched_name]["schedule_weeks"]
        ]

        for week in previous_week_names:
            # w = self.epjson["Schedule:Week:Daily"][week]
            # previous_day_names.extend(w.values())
            # Remove the week
            del self.epjson["Schedule:Week:Daily"][week]

        # previous_day_names = set(previous_day_names)
        # for day in previous_day_names:
        #     del self.epjson["Schedule:Day:Hourly"][day]

        # Add the new components
        self.epjson["Schedule:Year"][sched_name] = year_bunch
        for week in week_bunches:
            for name, values in week.items():
                self.epjson["Schedule:Week:Daily"][name] = values
        for day in day_bunches:
            for name, values in day.items():
                self.epjson["Schedule:Day:Hourly"][name] = values

    def handle_infiltration(self, template_dict):
        """
        Update infiltration specification
        """
        for infil_def in self.epjson["ZoneInfiltration:DesignFlowRate"].values():
            infil_def["design_flow_rate_calculation_method"] = "Flow/ExteriorArea"
            infil_def["flow_rate_per_exterior_surface_area"] = template_dict[
                "Infiltration"
            ]

    def handle_hvac_econ_enthalpy(self, template_dict):
        """
        Update Basic HVAC stuff
        # TODO: Should we also be updating the Sizing:Zone object?
        # limits are needed to avoid overheating - no limits, no stress (esp. for large shoeboxes) - cooling had large numbers, heating has no limit
        # OR look at sizing options for ideal loads?? Ask ben
        """
        for zone in self.epjson["ZoneHVAC:IdealLoadsAirSystem"].values():
            # Handle Heat Recovery
            if template_dict["RecoverySettings"] == HRV.NoHRV:
                zone["heat_recovery_type"] = "None"
            elif template_dict["RecoverySettings"] == HRV.Sensible:
                zone["sensible_heat_recovery_effectiveness"] = 0.7
                zone["heat_recovery_type"] = "Sensible"
            elif template_dict["RecoverySettings"] == HRV.Enthalpy:
                zone["sensible_heat_recovery_effectiveness"] = 0.7
                zone["latent_heat_recovery_effectiveness"] = 0.65
                zone["heat_recovery_type"] = "Enthalpy"

            # Handle Economizer
            zone["outdoor_air_economizer_type"] = Econ(
                template_dict["EconomizerSettings"]
            ).name

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
            self.epjson["Schedule:Constant"][schedule_name][
                "hourly_value"
            ] = template_dict["humid_max"]

        for schedule_name in humidimin_scheds_to_change:
            self.epjson["Schedule:Constant"][schedule_name][
                "hourly_value"
            ] = template_dict["humid_min"]

    def handle_ventilation(self, template_dict):
        """
        Update Ventilation

        If DCV is off, then the ventilation rate is set to the following:
        (floor_area * flow_per_area + flow_per_person * design_occ) * OA_schedule

        If DCV is on, then the ventilation rate is set to the following:
        (floor_area * flow_per_area + flow_per_person * design_occ * Occ_Schedule) * OA_schedule

        """

        mech_vent_sched_mode = MechVentMode(template_dict["VentilationMode"]).name

        logger.debug(f"Mechanical ventilation schedule: {mech_vent_sched_mode}")
        # TODO: check outputs of when ventilation is on
        self.epjson["DesignSpecification:OutdoorAir"]["SharedDesignSpecOutdoorAir"] = {
            "outdoor_air_flow_per_person": template_dict["VentilationPerPerson"],
            "outdoor_air_flow_per_zone_floor_area": template_dict["VentilationPerArea"],
            "outdoor_air_method": "Sum",
            "outdoor_air_schedule_name": "",  # AllOn
        }

        if mech_vent_sched_mode == MechVentMode.OccupancySchedule.name:
            logger.debug("Setting demand controlled ventilation to Occupancy Schedule.")
            for zone in self.epjson["ZoneHVAC:IdealLoadsAirSystem"].values():
                zone["demand_controlled_ventilation_type"] = mech_vent_sched_mode
        elif mech_vent_sched_mode == MechVentMode.AllOn.name:
            logger.debug("Setting demand controlled ventilation to AllOn.")
            for zone in self.epjson["ZoneHVAC:IdealLoadsAirSystem"].values():
                zone["demand_controlled_ventilation_type"] = "None"
        else:
            logger.debug("Setting demand controlled ventilation to Off.")
            for zone in self.epjson["ZoneHVAC:IdealLoadsAirSystem"].values():
                zone["demand_controlled_ventilation_type"] = "None"
            # make outdoor air schedule zero if mech vent mode is off
            self.epjson["DesignSpecification:OutdoorAir"]["SharedDesignSpecOutdoorAir"][
                "outdoor_air_schedule_name"
            ] = "Off"

    def handle_thermostat(self, template_dict):
        # TODO: allow for schedule?
        self.epjson["Schedule:Constant"]["CoolingSPSchedule"][
            "hourly_value"
        ] = template_dict["CoolingSetpoint"]
        self.epjson["Schedule:Constant"]["HeatingSPSchedule"][
            "hourly_value"
        ] = template_dict["HeatingSetpoint"]

    def handle_geometry(self):
        # scale geometry
        self.epjson = gu.scale_shoebox(
            sb=self.epjson,
            width=self.shoebox_config.width,
            height=self.shoebox_config.height,
            perim_depth=self.shoebox_config.perim_depth,
            core_depth=self.shoebox_config.core_depth,
        )
        # Update window to wall ratio
        self.epjson = gu.update_wwr(self.epjson, self.shoebox_config.wwr)
        # Change adiabatic roof and floor dimensions
        self.epjson = gu.set_adiabatic_surfaces(
            sb=self.epjson, shoebox_config=self.shoebox_config
        )

    def handle_shading(self):
        r = 2 * self.shoebox_config.width
        self.epjson = gu.build_shading(
            self.epjson,
            angles=self.shoebox_config.shading_vect,
            radius=r,
            override=False,
        )

    def idf(self, run_simulation=True):
        logger.info(f"Building idf for {self.ep_json_path}")
        idf_path = self.convert(path=self.ep_json_path)
        idf = IDF(idf_path, epw=self.epw, output_directory=self.output_directory)
        if run_simulation:
            hourly, monthly = self.simulate(idf)
            logger.info("HEATING/COOLING EUI")
            logger.info(monthly.sum() / self.floor_area * 2.77e-07)
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

    @classmethod
    def error_report(cls, idf):
        SEVERE = "** Severe  **"
        FATAL = "**  Fatal  **"
        WARNING = "** Warning **"
        NEXTLINE = "**   ~~~   **"
        filepath, *_ = idf.simulation_dir.files("*.err")
        errors = []
        warnings = []
        with open(filepath, "r") as f:
            for line in f:
                l = line.strip()
                if l.startswith(SEVERE) or l.startswith(FATAL):
                    errors.append(l)
                elif l.startswith(WARNING):
                    warnings.append(l)
        return (errors, warnings)

    def convert(self, path, file_type="idf"):
        logger.debug(f"Converting {path} to {file_type}")
        # Define the command and its arguments
        cmd = (
            settings.energyplus_location
            / f"energyplus{'.exe' if os.name == 'nt' else ''}"
        )
        args = ["--convert-only", "--output-directory", self.output_directory, path]

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

        return str(path).split(".")[0] + f".{file_type}"
        # TODO get actual new file name
        # return f"{self.name}.{file_type}"

    def compare_idfs(self, json_new):
        diff_report = jsondiff.diff(
            self._seed_epjson, json_new, syntax="symmetric", dump=True
        )
        if diff_report:
            fname = self.output_directory / f"{self.name}_changereport.json"
            json.dump(json.loads(diff_report), open(fname, "w"), indent=4)
            logger.info(f"Saved report as {fname}")
        else:
            logger.info("No changes in JSON found!")


if __name__ == "__main__":
    settings.energyplus_location = Path("C:\EnergyPlusV22-2-0")
    # settings.energyplus_location = Path("D:\EnergyPlusV22-2-0")
    settings.ep_version = "22.2.0"
    shoebox_config = ShoeboxConfiguration()
    shoebox_config.width = 10
    shoebox_config.height = 10
    shoebox_config.perim_depth = 5
    shoebox_config.core_depth = 4
    shoebox_config.adiabatic_partition_flag = 1
    # shoebox_config.floor_2_facade = 0.9
    # shoebox_config.core_2_perim = 0.4
    shoebox_config.roof_2_footprint = 0.25
    shoebox_config.ground_2_footprint = 0.5
    shoebox_config.wwr = 0.2
    shoebox_config.orientation = 0
    shoebox_config.shading_vect = np.random.random(SHADING_DIV_SIZE) * math.pi / 3

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

    epw = Path(
        "./ml-for-bem/data/epws/city_epws_indexed/cityidx_0001_USA_NY-New York Central Prk Obs Belv.725033_TMY3.epw"
    )
    out_dir = Path("./ml-for-bem/shoeboxer/cache")
    d = template_dict(
        schedules=scheds,
        PeopleDensity=0.05,
        LightingPowerDensity=2,
        EquipmentPowerDensity=5,
        Infiltration=0.0001,
        VentilationPerArea=0.0004,
        VentilationPerPerson=0.0025,
        VentilationMode=1,  # one hot of 0-2
        HeatingSetpoint=18,
        CoolingSetpoint=24,
        RecoverySettings=0,
        EconomizerSettings=0,
        FacadeRValue=2,
        FacadeMass=2,  # need to revisit minimums
        RoofRValue=6,
        RoofMass=1,
        SlabRValue=4,
        WindowShgc=0.5,
        WindowUValue=3.0,
    )

    sb = ShoeBox(
        name=f"test",
        seed_model=Path("./ml-for-bem/shoeboxer/shoebox-template.json"),
        shoebox_config=shoebox_config,
        epw=epw,
        output_directory=out_dir,
        template_dict=d,
        change_summary=True,
    )
    idf = sb.idf(run_simulation=False)
    hourly, monthly = sb.simulate(idf=idf)
    print("HEATING/COOLING EUI")
    print(monthly.sum() / sb.floor_area * 2.77e-07)
    # idf.view_model()
