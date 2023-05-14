import pandas as pd
import numpy as np
import random
import re

columns = [
    "bldg_id",  # numerical
    "archetype",  # residential type
    "vintage",  # year
    "climate_zone",
    "base_epw",  # city id
    "window_type",  # window id - index
    "wwr",  # as fraction
    "heating_setpoint",  # deg C
    "cooling_setpoint",  # deg C
    "Infiltration",  # ACH, converted from ACH50
    "dhw_flow",  # Flow rate per floor area
    "PeopleDensity",  # people/m2
    "LightingPowerDensity",  # W/m2 # TODO: REVIEW
    "EquipmentPowerDensity",  # W/m2
    "FacadeMass",  # J/(m2⋅K)
    # "RoofMass",
    "SlabMass",
    "FacadeRValue",
    "RoofRValue",
    "PartitionRValue",
    "SlabRValue",
    "has_electric_heating",  # t/f
    "heating_cop",  #
    "has_cooling",  # t/f
    "cooling_cop",  #
    # "has_recovery",
    # "recovery_type",
    # "recovery_eff",
    # "has_economizer",
]

# TODO: make it two mass scenarios, change to C and R-val, add high mass flag
# TODO: add value for construction types from base template

SQFT_TO_SQM = 0.092903
RVAL_TO_RSI = 1 / 5.678


FIBERGLASS_K = 0.043  # W
WOOD_STUD_K = 0.14  # J/kg.K
CMU_K = 1.25
BRICK_K = 0.41
CONCRETE_K = 1.75
TILE_K = 0.8

WOOD_STUD_T = 0.0381  # 2X4
CMU_T = 0.1524  # always 6 inches
BRICK_T = 0.3048  # always 3 wythe
CONCRETE_T = 0.1143  # 4-5 inches
TILE_T = 0.03

WOOD_STUD_C = 1200.0 * 0.10  # J/kg.K 2x4 16 o/c
CMU_C = 840.0
BRICK_C = 920.0
CONCRETE_C = 840.0
TILE_C = 840.0
FIBERGLASS_C = 840.0

WOOD_STUD_rho = 650.0  # kg/m3
CMU_rho = 880.0
BRICK_rho = 1000.0
CONCRETE_rho = 2400.0
TILE_rho = 2243.0
FIBERGLASS_rho = 12.0


def thermal_capacitance_per_area(c, rho, t):
    """
    Thermal capacitance for material with specific heat capacity, c (J/kg.K),
    density, rho (kg/m3), and thickness, t (m)
    """
    return c * rho * t  # J/(m2⋅K)


def F_TO_C(f):
    return (f - 32.0) * 5 / 9


CLIMATEZONES = {
    "0A": 0,
    "0B": 1,
    "1A": 2,
    "1B": 3,
    "2A": 4,
    "2B": 5,
    "3A": 6,
    "3B": 7,
    "3C": 8,
    "4A": 9,
    "4B": 10,
    "4C": 11,
    "5A": 12,
    "5B": 13,
    "5C": 14,
    "6A": 15,
    "6B": 16,
    "7A": 17,
    "8A": 18,
}

CLIMATEZONES_LIST = list(CLIMATEZONES.keys())


RESTYPES = {
    "Single-Family Detached": 0,
    "Single-Family Attached": 1,
    "Multi-Family with 2 - 4 Units": 2,
    "Multi-Family with 5+ Units": 3,
}

# TODO: replace numbers with archetypal template id
WINDTYPES = {
    "Single, Clear, Metal": 0,
    "Single, Clear, Non-metal": 0,
    "Single, Clear, Metal, Exterior Clear Storm": 1,
    "Single, Clear, Non-metal, Exterior Clear Storm": 1,
    "Double, Clear, Metal, Air": 1,
    "Double, Clear, Non-metal, Air": 1,
    "Double, Clear, Metal, Air, Exterior Clear Storm": 3,
    "Double, Clear, Non-metal, Air, Exterior Clear Storm": 3,
    "Double, Low-E, Non-metal, Air, M-Gain": 2,
    "Triple, Low-E, Non-metal, Air, L-Gain": 4,
}

LIGHTTYPES = {
    "100% Incandescent": 4.05,
    "100% LED": 0.7 * 4.05,  # 30% less than incandescent
    "100% CFL": 0.25 * 4.05,  # 75% less than incandescent
}

BASE_EPD = 5.38
BASE_DHW = 0.00049  # From boston residential template
# 3.66e-006 m3/s in NREL apartment

# HEATING = {
#     "Natural Gas Shared Heating",
#     "Natural Gas Fuel Boiler",
#     "Electricity ASHP",
#     "Natural Gas Fuel Furnace",
#     "Electricity Electric Furnace",
#     "Natural Gas Fuel Wall/Floor Furnace",
#     "Electricity Baseboard",
#     "Fuel Oil Shared Heating",
#     "Fuel Oil Fuel Boiler",
#     "Fuel Oil Fuel Furnace",
#     "Electricity Shared Heating",
#     "Fuel Oil Fuel Wall/Floor Furnace",
#     "Electricity Electric Wall Furnace",
# }


class ResStockConfiguration:
    """
    Class for configuring numeric vectors from ResStock samples
    """

    def __init__(self, resstock_samples, cities):
        """
        Create a ResStock paramter object

        Args:
            resstock_samples: pandas df of resstock outputs (from csv)
        Returns:
            samples_vector: np.ndarray, shape= (n, 20) for n samples
            epw_vector: np.ndarray, lookup table for epws

        """

        self.columns = columns

        self.samples = resstock_samples
        # print("IMPORTED SAMPLES")
        # print(self.samples.head())
        self.cities = cities
        self.clean()
        self.n = self.samples.shape[0]
        self.out = np.zeros(shape=(self.n, len(columns)))
        # TODO: return a pandas dataframe

    def clean(self):
        shp = self.samples.shape[0]
        self.samples = self.samples.drop(
            self.samples[self.samples["City"].str.contains("Balance")].index
        )
        shp_n = self.samples.shape[0]
        print(
            f"Dropped {shp-shp_n} samples while cleaning. Now there are {shp_n} samples."
        )

    def get_template_df(self):
        arr = self.get_template_vector()
        return pd.DataFrame(arr, columns=self.columns)

    def get_template_vector(self):
        self.apply_to_col(np.arange(self.n), "archetype")
        self.apply_to_col(self.get_type_idx(), "archetype")
        self.apply_to_col(self.get_age(), "vintage")
        # TODO: confirm format of city info
        self.apply_to_col(self.get_city_idx(), "base_epw")
        self.apply_to_col(self.get_climatezone(), "climate_zone")

        self.convert_internal_loads()
        self.convert_conditioning()
        self.convert_wall_construction()
        self.convert_roof_construction()
        self.convert_floor_construction()
        return self.out

    def convert_internal_loads(self):
        occupants = self.samples["Occupants"]

        def get_area(a):
            if "+" in a[0]:
                return int(a[0].split("+")[0])
            mm = a[0].split("-")
            mina = int(mm[0])
            maxa = int(mm[1])
            return random.randint(mina, maxa)

        areas = self.samples["Geometry Floor Area"].to_numpy()
        areas = areas.reshape((len(areas), 1))
        areas = np.apply_along_axis(get_area, 1, areas)
        areas = areas * SQFT_TO_SQM
        occ_density = occupants / areas
        self.apply_to_col(occ_density, "PeopleDensity")

        def get_perc(a):
            a = int(a.split("%")[0])
            return a / 100

        plg_loads = self.samples["Plug Loads"].apply(get_perc)
        plg_div = self.samples["Plug Load Diversity"].apply(get_perc)
        self.apply_to_col(plg_loads * plg_div * BASE_EPD, "EquipmentPowerDensity")

        dhw_loads = self.samples["Hot Water Fixtures"].apply(get_perc)
        self.apply_to_col(dhw_loads * BASE_DHW, "dhw_flow")

        def lookup_lpd(a):
            return LIGHTTYPES[a]

        lpd = self.samples["Lighting"].apply(lookup_lpd)
        self.apply_to_col(lpd, "LightingPowerDensity")

    def convert_conditioning(self):
        self.apply_to_col(self.convert_windows(), "window_type")
        self.apply_to_col(self.get_wwr(), "wwr")

        # Electricity = True, Gas = False
        fuel = self.samples["Heating Fuel"] == "Electricity"
        self.apply_to_col(fuel, "has_electric_heating")
        self.apply_to_col(
            self.get_temp(self.samples["Heating Setpoint"]), "heating_setpoint"
        )
        self.apply_to_col(self.heating_cop(), "heating_cop")

        self.apply_to_col(self.cooling_cop(), "cooling_cop")
        flag = self.samples["HVAC Cooling Type"] == "None"
        self.apply_to_col(~flag, "has_cooling")
        self.apply_to_col(np.ones(self.n) * 3.0, "cooling_cop")
        self.apply_to_col(
            self.get_temp(self.samples["Cooling Setpoint"]), "cooling_setpoint"
        )

        self.apply_to_col(self.calculate_ACH(), "Infiltration")

    def apply_to_col(self, data, name):
        if name not in self.columns:
            raise ValueError(f"Error indexing {name}")
        i = self.columns.index(name)
        self.out[:, i] = data

    def get_city_idx(self):  # TODO: citymap.json
        c = self.samples.filter(regex="City").merge(
            self.cities, left_on="City", right_on="city", how="left"
        )
        return c["index"]

    def get_climatezone(self):
        def lookup_cz(a):
            return CLIMATEZONES[a]

        return self.samples["ASHRAE IECC Climate Zone 2004"].apply(lookup_cz)

    def get_type_idx(self):
        def lookup_type(a):
            return RESTYPES[a]

        return self.samples["Geometry Building Type RECS"].apply(lookup_type)

    def get_age(self):
        def get_age(a):
            if "<" in a:
                return random.randint(1800, 1940)
            a = int(a.split("s")[0])
            return random.randint(a, a + 10)

        return self.samples["Vintage"].apply(get_age)

    def convert_windows(self):
        def lookup_window(a):
            return WINDTYPES[a]

        return self.samples["Windows"].apply(lookup_window)

    def get_wwr(self):
        def convert_wwr(a):
            a = int(re.findall(r"\d+", a)[0])
            return a / 100

        return self.samples["Window Areas"].apply(convert_wwr)

    def get_temp(self, array):
        def convert_temp(a):
            a = int(a.split("F")[0])
            return F_TO_C(a)

        return array.apply(convert_temp)

    def calculate_ACH(self):
        def get_ach(a):
            a = int(a.split(" ACH50")[0])
            return a / 15.0  # TODO: discuss method

        return self.samples["Infiltration"].apply(get_ach)

    def cooling_cop(self):
        def get_cop(a):
            if a == "Heat Pump":
                return 4
            if "SEER" in a:
                seer = int(re.findall(r"\d+", a)[0])
                return seer * 1055 / 3600
            if "EER" in a:
                eer = float(re.findall(r"\d+\.\d*", a)[0])
                return eer / 3.41214
            if a == "Shared Cooling":
                return 3  # TODO
            else:
                return 0

        return self.samples["HVAC Cooling Efficiency"].apply(get_cop)

    def heating_cop(self):
        def get_cop(a):
            if "AFUE" in a:
                a = int(re.findall(r"\d+", a)[0])
                return a / 100
            if "SEER" in a:  # HSPF???
                seer = int(re.findall(r"\d+", a)[0])
                return seer * 1055 / 3600
            if a == "Shared Heating":
                return 0.9  # TODO
            else:
                return 0

        return self.samples["HVAC Heating Efficiency"].apply(get_cop)

    def get_mass(self, a):  # TODO fix the roof mass
        if "CMU" in a:
            c = thermal_capacitance_per_area(CMU_C, CMU_rho, CMU_T)
            r = CMU_T / CMU_K
        elif "Brick" in a:
            c = thermal_capacitance_per_area(BRICK_C, BRICK_rho, BRICK_T)
            r = BRICK_T / BRICK_K
        elif "Concrete" in a:
            c = thermal_capacitance_per_area(CONCRETE_C, CONCRETE_rho, CONCRETE_T)
            r = CONCRETE_T / CONCRETE_K
        elif "Tile" in a or "Slate" in a:
            c = thermal_capacitance_per_area(TILE_C, TILE_rho, TILE_T)
            r = TILE_T / TILE_K
        else:
            # elif "Wood" in a:
            c = thermal_capacitance_per_area(WOOD_STUD_C, WOOD_STUD_rho, WOOD_STUD_T)
            r = WOOD_STUD_T / WOOD_STUD_K
        # else:
        # raise ValueError("Material not supported", a)
        return pd.Series([r, c])

    def get_insulation(self, a):
        if "R" in a:
            # ins = int(a.split("R-")[1])
            ins = int(re.findall(r"R.(\d+)", a)[0])
            r = ins * RVAL_TO_RSI
            t = r * FIBERGLASS_K
            c = thermal_capacitance_per_area(FIBERGLASS_C, FIBERGLASS_rho, t)
        else:
            r = 0
            c = 0
        return pd.Series([r, c])

    def convert_wall_construction(self):
        ins_layer = self.samples["Insulation Wall"].apply(self.get_insulation)
        # print(self.samples["Insulation Wall"].head())
        # print("\nTHIS IS INSULATION LAYER\n", ins_layer)
        mass_layer = self.samples["Insulation Wall"].apply(self.get_mass)
        # print("\nTHIS IS mass LAYER\n", mass_layer)
        ins = ins_layer[0] + mass_layer[0]
        mass = ins_layer[1] + mass_layer[1]
        self.apply_to_col(ins, "FacadeRValue")
        self.apply_to_col(mass_layer[1], "FacadeMass")

    def convert_roof_construction(self):
        # If no insulation in roof, use ceiling insulation
        ins_layer = []
        for i, a in enumerate(self.samples["Insulation Roof"]):
            if "R" in a:
                ins_layer.append(a)
            else:
                ins_layer.append(self.samples["Insulation Roof"].iloc[i])

        ins_layer = (
            pd.Series(ins_layer).apply(self.get_insulation).reset_index(drop=True)
        )
        mass_layer = (
            self.samples["Roof Material"].apply(self.get_mass).reset_index(drop=True)
        )
        ins = ins_layer[0] + mass_layer[0]
        mass = ins_layer[1] + mass_layer[1]
        self.apply_to_col(ins, "RoofRValue")
        # self.apply_to_col(mass, "RoofMass") # TODO: take out

    def convert_floor_construction(self):
        # ins_t = []
        # for i, a in enumerate(self.samples["Insulation Floor"]):
        #     if "R-" in a:
        #         ins_t.append(self.get_insulation(a))
        #     else:
        #         if "R" in a:
        #             ins_t.append(int(re.findall(r"ft R(\d+)", a)[0]))
        #         else:
        #             ins_t.append(None)
        mass_layer = self.samples["Insulation Roof"].apply(self.get_mass)
        ins_layer = self.samples["Insulation Roof"].apply(self.get_insulation)
        ins = ins_layer[0] + mass_layer[0]
        mass = ins_layer[1] + mass_layer[1]
        self.apply_to_col(ins, "SlabRValue")
        self.apply_to_col(mass, "SlabMass")


if __name__ == "__main__":
    filepath = "C:/Users/zoele/Git_Repos/ml-for-building-energy-modeling/NREL/data/ResStock/resstock_filtered.csv"
    citypath = "C:/Users/zoele/Git_Repos/ml-for-building-energy-modeling/NREL/data/ResStock/cities.csv"
    resstock_filtered = pd.read_csv(filepath, index_col=0)
    cities = pd.read_csv(citypath)
    test = ResStockConfiguration(resstock_filtered, cities)
    # test.get_template_vector()
    df = test.get_template_df()
    # print(test.out[0])
    print(df.head())
    print(df.describe()["FacadeMass"])
    print(resstock_filtered.describe())
    print(test.out.shape)
