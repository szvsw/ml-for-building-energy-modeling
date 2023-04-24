import pandas as pd
import numpy as np
import random
import re

columns = [
    "bldg_id",  # numerical
    "archetype",  # residential type
    "vintage",  # year
    # "template_id", #TODO
    "epw_id",  # city id
    "window_type",  # window id
    "wwr",  # as fraction
    "has_electric_heating",  # t/f
    "heating_cop",  #
    "heating_setpoint",  # deg C
    "has_cooling",  # t/f
    "cooling_cop",  #
    "cooling_setpoint",  # deg C
    # "has_recovery",
    # "recovery_type",
    # "recovery_eff",
    # "has_economizer",
    "ach",  # Converted from ACH50
    "dhw_flow",  # Flow rate per floor area
    "people_density",  # people/m2
    "epd",  # W/m2
    "lpd",  # W/m2
]

SQFT_TO_SQM = 0.092903


def F_TO_C(f):
    return (f - 32.0) * 5 / 9


RESTYPES = {
    "Single-Family Detached": 0,
    "Single-Family Attached": 1,
    "Multi-Family with 2 - 4 Units": 2,
    "Multi-Family with 5+ Units": 3,
}

WINDTYPES = {
    "Single, Clear, Metal": 0,
    "Single, Clear, Non-metal": 1,
    "Single, Clear, Metal, Exterior Clear Storm": 2,
    "Single, Clear, Non-metal, Exterior Clear Storm": 3,
    "Double, Clear, Metal, Air": 4,
    "Double, Clear, Non-metal, Air": 5,
    "Double, Clear, Metal, Air, Exterior Clear Storm": 6,
    "Double, Clear, Non-metal, Air, Exterior Clear Storm": 7,
    "Double, Low-E, Non-metal, Air, M-Gain": 8,
    "Triple, Low-E, Non-metal, Air, L-Gain": 9,
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
        self.samples = resstock_samples
        self.clean()
        self.n = self.samples.shape[0]
        self.out = np.zeros(shape=(self.n, len(columns)))

    def clean(self):
        shp = self.samples.shape[0]
        self.samples = self.samples.drop(
            self.samples[self.samples["City"].str.contains("Balance")].index
        )
        shp_n = self.samples.shape[0]
        print(
            f"Dropped {shp-shp_n} samples while cleaning. Now there are {shp_n} samples."
        )

    def get_template_vector(self):
        self.apply_to_col(np.arange(self.n), "archetype")
        self.apply_to_col(self.get_type_idx(), "archetype")
        self.apply_to_col(self.get_age(), "vintage")
        # TODO: confirm format of city info
        self.apply_to_col(self.get_city_idx(cities), "epw_id")

        self.convert_internal_loads()
        self.convert_conditioning()
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
        self.apply_to_col(occ_density, "people_density")

        def get_perc(a):
            a = int(a.split("%")[0])
            return a / 100

        plg_loads = self.samples["Plug Loads"].apply(get_perc)
        plg_div = self.samples["Plug Load Diversity"].apply(get_perc)
        self.apply_to_col(plg_loads * plg_div * BASE_EPD, "epd")

        dhw_loads = self.samples["Hot Water Fixtures"].apply(get_perc)
        self.apply_to_col(dhw_loads * BASE_DHW, "epd")

        def lookup_lpd(a):
            return LIGHTTYPES[a]

        lpd = self.samples["Lighting"].apply(lookup_lpd)
        self.apply_to_col(lpd, "lpd")

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

        self.apply_to_col(self.calculate_ACH(), "ach")

    def apply_to_col(self, data, name):
        if name not in columns:
            raise ValueError
        i = columns.index(name)
        self.out[:, i] = data

    def get_city_idx(self, cities):
        c = self.samples.filter(regex="City").merge(
            cities, left_on="City", right_on="city", how="left"
        )
        return c["index"]

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


if __name__ == "__main__":
    filepath = "C:/Users/zoele/Git_Repos/ml-for-building-energy-modeling/NREL/data/ResStock/resstock_filtered.csv"
    citypath = "C:/Users/zoele/Git_Repos/ml-for-building-energy-modeling/NREL/data/ResStock/cities.csv"
    resstock_filtered = pd.read_csv(filepath, index_col=0)
    cities = pd.read_csv(citypath)
    test = ResStockConfiguration(resstock_filtered, cities)
    test.get_template_vector()
    print(test.out[0])
    print(test.out.shape)
