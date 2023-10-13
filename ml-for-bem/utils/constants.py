import os
import sys
from pathlib import Path
from enum import IntEnum

JOULES_TO_KWH = 2.777e-7
SHADING_DIV_SIZE = 12

EPW_RELATIVE_PATH = "data/epws/city_epws_indexed"
SHOEBOX_RELATIVE_PATH = "shoeboxer/cache"
HIGH_LOW_MASS_THRESH = 00000  # J/m2K # TODO why is this zero

WINDOW_TYPES = {
    0: "single_clr",
    1: "Double_clr",
    2: "dbl_LoE",
    3: "triple_clr",
    4: "triple_LoE",
}


class ThermalMassConstructions(IntEnum):
    Concrete = 0
    Brick = 1
    WoodFrame = 2
    SteelFrame = 3


class ThermalMassCapacities(IntEnum):  # TODO
    Concrete = 450000
    Brick = 100000
    WoodFrame = 50000
    SteelFrame = 20000


class HRV(IntEnum):
    NoHRV = 0
    Sensible = 1
    Enthalpy = 2


class Econ(IntEnum):
    NoEconomizer = 0
    DifferentialEnthalpy = 1
    # DifferentialDryBulb = 2


class MechVentMode(IntEnum):
    Off = 0
    AllOn = 1
    OccupancySchedule = 2


class BooleanParam(IntEnum):
    false = 0
    true = 1


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

sched_type_limits = dict(
    key="SCHEDULETYPELIMITS",
    Name="Fraction",
    Lower_Limit_Value=0.0,
    Upper_Limit_Value=1.0,
    Numeric_Type="Continuous",
    Unit_Type="Dimensionless",
)


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
        freq="Hourly",
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
