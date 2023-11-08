import os
import json
from glob import glob

from functools import reduce
from pathlib import Path
from typing import List
import collections

import numpy as np
import math

import matplotlib.pyplot as plt
import logging

logging.basicConfig()
logger = logging.getLogger("Schema")
logger.setLevel(logging.INFO)

try:
    from archetypal.template.materials.material_layer import MaterialLayer
except (ImportError, ModuleNotFoundError) as e:
    logger.error("Failed to import a package! Be wary about continuing...", exc_info=e)

from utils.constants import *
from utils.schedules import (
    schedule_paths,
    operations,
    get_schedules,
    mutate_timeseries,
    update_schedule_objects,
)

data_path = Path(os.path.dirname(os.path.abspath(__file__))) / "data"

constructions_lib_path = os.path.join(
    os.getcwd(),
    "ml-for-bem",
    "data",
    "template_libs",
    "ConstructionsLibrary.json",
)


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
                    else np.array([value])
                )
                onehots = np.zeros((counts.shape[0], self.count))
                onehots[np.arange(counts.shape[0]), counts[:].astype(int)] = 1
                return onehots
            elif isinstance(self, SchedulesParameters):
                return (
                    self.extract_storage_values_batch(storage_batch)
                    if value is None
                    else value
                )
            else:
                vals = self.normalize(
                    self.extract_storage_values_batch(storage_batch).reshape(
                        -1, *self.shape_ml
                    )
                    if value is None
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

    def mutate_simulation_object(self):
        """
        This method updates the simulation objects (archetypal template, shoebox config)
        by extracting values for this parameter from the sim's storage vector and using this
        parameter's logic to update the appropriate objects.
        The default base SchemaParameter does nothing.  Children classes implement the appropriate
        semantic logic.
        Args:
        """
        pass

    def extract_from_template(self):
        """
        This method extracts the parameter value from an archetypal building template for the creation of a building vector.
        Works as the reverse of mutate_simulation_object
        Args:
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


class BuildingTemplateParameter(NumericParameter):
    __slots__ = "path"

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = path.split(".")


class RValueParameter(BuildingTemplateParameter):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)


class TMassParameter(OneHotParameter):
    __slots__ = "path"

    def __init__(self, path, **kwargs):
        super().__init__(count=4, **kwargs)
        self.path = path.split(".")

    def get_tmas_idx(self, val):
        if val >= ThermalMassCapacities.Concrete:
            hot_bin = ThermalMassConstructions.Concrete.value
        elif (
            val < ThermalMassCapacities.Concrete and val >= ThermalMassCapacities.Brick
        ):
            hot_bin = ThermalMassConstructions.Brick.value
        elif (
            val < ThermalMassCapacities.Brick and val >= ThermalMassCapacities.WoodFrame
        ):
            hot_bin = ThermalMassConstructions.WoodFrame.value
        elif val < ThermalMassCapacities.WoodFrame:
            hot_bin = ThermalMassConstructions.SteelFrame.value
        return hot_bin


class WindowParameter(NumericParameter):
    def __init__(self, min, max, **kwargs):
        super().__init__(min=min, max=max, shape_storage=(1,), shape_ml=(1,), **kwargs)


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
                    min=2,
                    max=8,
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
                    name="perim_depth",  # PERIM DEPTH, CORE DEPTH, GROUND DEPTH, ROOF DEPTH
                    min=1.25,
                    max=12,
                    mean=5,
                    std=0.25,
                    source="new",
                    info="Peimeter depth [m]",
                ),
                ShoeboxGeometryParameter(
                    name="core_depth",
                    min=1.25,
                    max=60,
                    mean=5,
                    std=0.25,
                    source="new",
                    info="Core to Perimeter ratio (unitless)",
                ),
                ShoeboxGeometryParameter(
                    name="roof_2_footprint",
                    min=0.0,
                    max=1.00,
                    mean=0.5,
                    std=0.25,
                    source="dogan_shoeboxer_2017",
                    info="Roof to footprint ratio (unitless)",
                ),
                ShoeboxGeometryParameter(
                    name="ground_2_footprint",
                    min=0.00,
                    max=1.0,
                    mean=0.5,
                    std=0.25,
                    source="dogan_shoeboxer_2017",
                    info="Ground to footprint ratio (unitless)",
                ),
                ShoeboxGeometryParameter(
                    name="orientation",
                    min=0.0,
                    max=2 * np.pi,
                    mean=np.pi,
                    std=0.25,
                    info="Orientation",
                ),
                ShoeboxGeometryParameter(
                    name="wwr",
                    min=0.0,
                    max=0.9,
                    mean=0.3,
                    std=0.25,
                    info="Window-to-wall Ratio (unitless)",
                ),
                # ShoeboxOrientationParameter(
                #     name="orientation",
                #     info="Shoebox Orientation",
                # ),
                BuildingTemplateParameter(
                    name="HeatingSetpoint",
                    path="Conditioning.HeatingSetpoint",
                    min=14,
                    max=24,
                    mean=21,
                    std=2,
                    info="Heating setpoint",
                ),
                BuildingTemplateParameter(
                    name="CoolingSetpoint",
                    path="Conditioning.CoolingSetpoint",
                    min=22,
                    max=30,
                    mean=24,
                    std=1,
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
                    max=30,
                    mean=10,
                    std=6,
                    source="ComStock",
                    info="Lighting Power Density [W/m2]",
                ),
                BuildingTemplateParameter(
                    name="EquipmentPowerDensity",
                    path="Loads.EquipmentPowerDensity",
                    min=0,
                    max=60,  # TODO this is foor super high density spaces (like mech rooms). Alternative is 500
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
                    min=0.0,
                    max=0.001,
                    mean=0.0006,
                    std=0.0002,
                    source="tacit",
                    info="Infiltration rate [m3/s/m2 ext area]",
                ),
                BuildingTemplateParameter(
                    name="VentilationPerArea",
                    path="Conditioning.MinFreshAirPerArea",  # TODO check & set max and min
                    min=0.0,
                    max=0.005,
                    mean=0.0004,
                    std=0.0002,
                    source="tacit",
                    info="Outdoor air flow per floor area, minimum (m3/s/m2)",
                ),
                BuildingTemplateParameter(
                    name="VentilationPerPerson",
                    path="Conditioning.MinFreshAirPerPerson",  # TODO check & set max and min
                    min=0,
                    max=0.015,
                    mean=0.002,
                    std=0.001,
                    source="tacit",
                    info="Outdoor air flow per person, minimum (m3/s/person)",
                ),
                OneHotParameter(
                    name="VentilationMode",
                    count=3,
                    info="Mode setter for mechanical ventilation response schedule",
                ),
                TMassParameter(
                    name="FacadeMass",
                    path="Facade",
                    source="https://www.designingbuildings.co.uk/",
                    info="Exterior wall thermal mass (J/Km2)",
                ),
                TMassParameter(
                    name="RoofMass",
                    path="Roof",
                    source="https://www.designingbuildings.co.uk/",
                    info="Exterior wall thermal mass (J/Km2)",
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
                    name="WindowUValue",
                    min=0.3,
                    max=7.0,
                    mean=5.0,
                    std=2.0,
                    source="climate studio",
                    info="U-value (m2K/W)",
                ),
                WindowParameter(
                    name="WindowShgc",
                    min=0.05,
                    max=0.99,
                    mean=0.5,
                    std=0.1,
                    source="climate studio",
                    info="SHGC",
                ),
                OneHotParameter(
                    name="EconomizerSettings",
                    count=2,
                    info="Flag for economizer use.",
                ),
                OneHotParameter(
                    name="RecoverySettings",
                    count=3,
                    info="Index for use of heat recovery (type) - none, hrv, erv.",
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
                freq="Hourly",
                store_output=True,
            ),
            TimeSeriesOutput(
                name="Cooling",
                key="OUTPUT:VARIABLE",
                var_name="Zone Ideal Loads Zone Total Cooling Energy",
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

    def __getitem__(self, key) -> SchemaParameter:
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
                if isinstance(parameter, SchedulesParameters):
                    schedules_vect = parameter.extract_from_template(building_template)
                elif isinstance(parameter, TMassParameter):
                    vals = parameter.extract_from_template(building_template)
                    # append to template vector
                    template_vect.extend(
                        vals[0]
                    )  # Never occurs for batch (one template at a time)
                elif (
                    isinstance(parameter, BuildingTemplateParameter)
                    or isinstance(parameter, RValueParameter)
                    or isinstance(parameter, WindowParameter)
                ):
                    val = parameter.extract_from_template(building_template)
                    # append to template vector
                    template_vect.append(val)

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
