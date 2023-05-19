import numpy as np
import pandas as pd
import os

from pyumi.epw import Epw
from ladybug.epw import EPWFields

from pvlib import irradiance
from pvlib.location import Location
from pvlib.solarposition import declination_spencer71

# Check that all epws have full datapoints (interpolate if nulls)
"""
*   0 Year
*   1 Month
*   2 Day
*   3 Hour
*   4 Minute
*   5 Uncertainty Flags
*   6 Dry Bulb Temperature
*   7 Dew Point Temperature
*   8 Relative Humidity
*   9 Atmospheric Station Pressure
*   10 Extraterrestrial Horizontal Radiation
*   11 Extraterrestrial Direct Normal Radiation
*   12 Horizontal Infrared Radiation Intensity
*   13 Global Horizontal Radiation
*   14 Direct Normal Radiation
*   15 Diffuse Horizontal Radiation
*   16 Global Horizontal Illuminance
*   17 Direct Normal Illuminance
*   18 Diffuse Horizontal Illuminance
*   19 Zenith Luminance
*   20 Wind Direction
*   21 Wind Speed
*   22 Total Sky Cover
*   23 Opaque Sky Cover
*   24 Visibility
*   25 Ceiling Height
*   26 Present Weather Observation
*   27 Present Weather Codes
*   28 Precipitable Water
*   29 Aerosol Optical Depth
*   30 Snow Depth
*   31 Days Since Last Snowfall
*   32 Albedo
*   33 Liquid Precipitation Depth
*   34 Liquid Precipitation Quantity
"""

fields = [6, 8, 9, 13, 14, 15]
n_channels = len(fields) + 1


class RadFacade:
    """
    Adapted from Alpha Arsano's ClimaBox and Sam D. code to calculate facade radiation for exterior surface.
    """

    def __init__(
        self,
        timezone,
        latitude,
        longitude,
        city,
        GlobHzRad,
        DirNormRad,
        DiffRad,
        datetime,
        solar_position_method="nrel_numpy",
    ):
        """
        Init with weather in pd database format with ['ghi', 'dni', 'dhi']
        Location data (city, lat, long, time zone)
        solar_position_method (str): default 'nrel_numpy'.
                'nrel_numpy' uses an implementation of the NREL SPA algorithm.
                'nrel_numba' uses an implementation of the NREL SPA algorithm, but also compiles the code first.
                'pyephem' uses the PyEphem package.
                'ephemeris' uses the pvlib ephemeris code. 'nrel_c' uses the NREL SPA C code.
        """

        pv_tz = f"Etc/GMT{int(timezone):+d}"

        self.location = Location(
            latitude,
            longitude,
            name=city,
            tz=pv_tz,
        )
        self.solar_position_method = solar_position_method

        weather = pd.DataFrame(columns=["ghi", "dni", "dhi"])
        weather["ghi"] = GlobHzRad
        weather["dni"] = DirNormRad
        weather["dhi"] = DiffRad
        idx = pd.DatetimeIndex(datetime).tz_localize(self.location.tz)
        weather = weather.set_index(idx)
        self.weather = weather

        self.declination = np.vectorize(declination_spencer71)(
            self.weather.index.dayofyear
        )
        self.solar_position = self.location.get_solarposition(
            self.weather.index, method=self.solar_position_method
        )
        self.clearsky = self.location.get_clearsky(self.weather.index)

    @property
    def azimuth(self):
        return self.solar_position["azimuth"].values

    @property
    def location(self):
        """Get or set the location."""
        return self._location

    @location.setter
    def location(self, value):
        assert isinstance(value, Location)
        self._location = value

    @property
    def weather(self) -> pd.DataFrame:
        """Get or set the weather DataFrame.
        Note: columns are ``ghi, dni, dhi``.
        """
        return self._weather

    @weather.setter
    def weather(self, value):
        assert isinstance(value, pd.DataFrame), "weather must be a DataFrame."
        assert sorted(value.columns.tolist()) == sorted(["ghi", "dni", "dhi"]), (
            f"columns of weather must be '['ghi', 'dni', 'dhi']', "
            f"not '{value.columns.tolist()}'"
        )
        assert isinstance(
            value.index, pd.DatetimeIndex
        ), f"weather.index must be a '{pd.DatetimeIndex}', not a '{type(value.index)}'"
        assert (
            value.index.tzinfo is not None
        ), "weather.index must be localized. use 'pandas.DatetimeIndex.tz_localize'."
        self._weather = value

    @property
    def solar_position_method(self):
        """Get or set the solar position method."""
        return self._solar_position_method

    @solar_position_method.setter
    def solar_position_method(self, value):
        assert value in [
            "nrel_numpy",
            "nrel_numba",
            "pyephem",
            "ephemeris",
            "nrel_c",
        ], (
            "Input value error for '{value}'. solar_position_method must be one of ("
            "'nrel_numpy', 'nrel_numba', 'pyephem', 'ephemeris', 'nrel_c')"
        )
        self._solar_position_method = value

    @property
    def solar_position(self):
        """Calculate the solar zenith, azimuth, etc. at this location."""
        return self._solar_position

    @solar_position.setter
    def solar_position(self, value):
        self._solar_position = value

    def get_irradiance_for_surface(self, tilt, surface_azimuth, times=None):
        """Calculate clear-sky GHI and transpose to plane of array Define a function
        so that we can re-use the sequence of operations with different locations.
        Args:
            tilt (float):
            surface_azimuth (float): surface azimuth from north.
        Returns:
            pd.DataFrame: total_irrad - Contains keys/columns 'poa_global',
                'poa_direct', 'poa_diffuse', 'poa_sky_diffuse', 'poa_ground_diffuse'.
        """
        if times is None:
            times = self.weather.index

        # Get weather data using values for GHI, DNI, and DHI
        # weather = self.weather.loc[times, :]
        weather = self.weather.loc[times, :]
        # Get solar azimuth and zenith to pass to the transposition function
        solar_position = self.solar_position.loc[times, :]
        # Use the get_total_irradiance function to transpose the GHI to POA
        poa_irradiance = irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=surface_azimuth,
            dni=weather["dni"],
            ghi=weather["ghi"],
            dhi=weather["dhi"],
            solar_zenith=solar_position["zenith"],
            solar_azimuth=solar_position["azimuth"],
        )
        # Return DataFrame

        return poa_irradiance  # Units are W/m2


def calc_surface_temp(epw_obj, orientation=0, refl_wall=0.026):
    _rad_obj = RadFacade(
        timezone=epw_obj.metadata["time-zone"],
        latitude=epw_obj.location.latitude,
        longitude=epw_obj.location.longitude,
        city=epw_obj.metadata["city"],
        GlobHzRad=epw_obj.global_horizontal_radiation.values,
        DirNormRad=epw_obj.direct_normal_radiation.values,
        DiffRad=epw_obj.diffuse_horizontal_radiation.values,
        datetime=epw_obj.dry_bulb_temperature.datetimes,
    )

    _, _difrad = (
        _rad_obj.get_irradiance_for_surface(tilt=90, surface_azimuth=orientation)
        .loc[:, ["poa_direct", "poa_diffuse"]]
        .values.T
    )
    facade_tot_irr = np.add(_difrad, _difrad)  # Units W/m2

    # Sol-air Temp = T_o + a * tot_rad - b
    # Assume dark surfaces
    b = 4.0  # wall, b=0 for roof

    t_sa = [
        x + refl_wall * y - b
        for x, y in zip(epw_obj.dry_bulb_temperature.values, facade_tot_irr)
    ]
    return t_sa


def collect_values(epw_obj):
    # epw_obj.dry_bulb_temperature.values
    # epw_obj.relative_humidity.values
    # epw_obj.atmospheric_station_pressure.values
    # epw_obj.global_horizontal_radiation.values
    # epw_obj.direct_normal_radiation.values
    # epw_obj.diffuse_horizontal_radiation.values
    value_array = np.zeros((n_channels, 8760))
    missing = {}
    for i, field_number in enumerate(fields):
        clim_values = epw_obj._get_data_by_field(field_number).values
        field = EPWFields.field_by_number(field_number)
        mis_val = field.missing if field.missing is not None else 0
        missing_idxs = []
        for idx, val in enumerate(clim_values):
            if val == mis_val:
                missing_idxs.append(idx)
        if len(missing_idxs) > 0:
            missing[field_number] = {
                "total_missing": len(missing_idxs),
                "missing_idxs": missing_idxs,
            }
        value_array[i, :] = clim_values
    if len(missing) > 0:
        raise ValueError("Missing values: ", missing)
        # Replace missing
        # epw_obj._data[field_number].append(mis_val)

    value_array[i + 1] = epw_obj.sky_temperature.values  # derived from radiation
    return value_array


if __name__ == "__main__":
    # ml-for-bem\data\epws\city_epws_indexed
    epw_base_path = os.path.join(
        os.getcwd(), "ml-for-bem", "data", "epws", "city_epws_indexed"
    )
    epw_path_list = os.listdir(epw_base_path)
    epw_path_list = [x for x in epw_path_list if ".epw" in x]

    # refl_wall = 0.5  # TODO: CHECK THIS

    climarray = np.zeros((len(epw_path_list), n_channels, 8760))
    print("Building climate lookup array of shape ", climarray.shape)

    for i, epw_path in enumerate(epw_path_list):
        print("Processing ", epw_path)
        epw_obj = Epw(os.path.join(epw_base_path, epw_path))
        climarray[i] = collect_values(epw_obj)
        # t_sa = [
        #     x + refl_wall * y - b for x, y in zip(climate.dbt, facade_tot_irr)
        # ]
    # hdf5
    np.save(
        os.path.join(os.getcwd(), "ml-for-bem", "data", "epws", f"climate_array.npy"),
        climarray,
    )
    # n weather files x channels x 8760
    # n weather files x 4 (tsol) x 8760
