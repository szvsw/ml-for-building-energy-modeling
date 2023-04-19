from functools import reduce

import numpy as np

from schedules import schedule_paths, operations

class SchemaParameter:
    __slots__ = (
        "name",
        "target_object_key",
        "dtype",
        "start_storage",
        "start_ml",
        "shape_storage",
        "shape_ml",
        "len_storage",
        "len_ml",
        "in_ml"
    )

    def __init__(self, name, shape_storage=(1,), shape_ml=(1,), target_object_key=None, dtype="scalar"):
        self.name = name
        self.target_object_key = target_object_key 
        self.dtype = dtype

        self.shape_storage = shape_storage
        self.shape_ml = shape_ml
        if shape_ml == (0,):
            self.in_ml = False

        self.len_storage = reduce(lambda a,b: a*b, shape_storage)
        self.len_ml = reduce(lambda a,b: a*b, shape_ml)

    def extract_values(self, storage_vector):
        data = storage_vector[self.start_storage:self.start_storage+self.len_storage]
        if self.shape_storage == (1,):
            return data[0]
        else:
            return data.reshape(*self.shape)
    
    def extract_values_batch(self, storage_batch):
        data = storage_batch[:,self.start_storage:self.start_storage+self.len_storage]
        return data.reshape(-1,*self.shape)
    
    def normalize(self, val):
        return val

    def unnormalize(self, val):
        return val

    
    def mutate_simulation_objects(self, epw, template, shoebox_dict):
        pass

class NumericParameter(SchemaParameter):
    __slots__ = (
        "min",
        "max",
        "range"
    )
    def __init__(self, min=0, max=1, **kwargs):
        super().__init__(**kwargs)
        self.min = min
        self.max = max
        self.range = self.max - self.min
    
    def normalize(self, value):
        return (value - self.min)/self.range
    
    def unnormalize(self, value):
        return value*self.range + self.min

class OneHotParameter(SchemaParameter):
    __slots__ = (
        "count"
    )

    def __init__(self, count, **kwargs):
        super().__init__(dtype="onehot", shape_ml=(count,), **kwargs)
        self.count = count
    
class ShoeboxGeometryParameter(NumericParameter):
    __slots__ = (

    )
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ShoeboxOrientationParameter(OneHotParameter):
    __slots__ = (

    )
    def __init__(self, **kwargs):
        super().__init__(count=4,**kwargs)

class BuildingTemplateParameter(NumericParameter):
    __slots__ = (
        "path"
    )
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = path

class RValueParameter(BuildingTemplateParameter):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

class SchedulesParameters(SchemaParameter):
    __slots__ = (
    )
    paths = schedule_paths
    operations = operations
    def __init__(self, **kwargs):
        super().__init__(name="schedules", shape_storage=(len(schedule_paths),len(operations)), shape_ml=(len(schedule_paths),8760), **kwargs)


    

class Schema:
    __slots__ = (
        "parameters",
        "storage_vec_len",
        "ml_vec_len",
        "_key_ix_lookup"
    )

    def __init__(self):
        self.parameters = [
            SchemaParameter(
                name="id",
                dtype="index",
                shape_ml=(0,)
            ),
            SchemaParameter(
                name="base_template",
                dtype="index",
                shape_ml=(0,)
            ),
            SchemaParameter(
                name="base_epw",
                dtype="index",
                shape_ml=(0,)
            ),
            ShoeboxGeometryParameter(
                name="width",
                min=1.5,
                max=5
            ),
            ShoeboxGeometryParameter(
                name="height",
                min=2.5,
                max=5
            ),
            ShoeboxGeometryParameter(
                name="facade_2_footprint",
                min=0.5,
                max=5
            ),
            ShoeboxGeometryParameter(
                name="perim_2_footprint",
                min=0.5,
                max=5
            ),
            ShoeboxGeometryParameter(
                name="roof_2_footprint",
                min=0.5,
                max=5
            ),
            ShoeboxGeometryParameter(
                name="footprint_2_ground",
                min=0.5,
                max=5
            ),
            ShoeboxGeometryParameter(
                name="shading_fact",
                min=0,
                max=1
            ),
            ShoeboxGeometryParameter(
                name="wwr_n",
                min=0,
                max=1
            ),
            ShoeboxGeometryParameter(
                name="wwr_e",
                min=0,
                max=1
            ),
            ShoeboxGeometryParameter(
                name="wwr_s",
                min=0,
                max=1
            ),
            ShoeboxGeometryParameter(
                name="wwr_w",
                min=0,
                max=1
            ),
            ShoeboxOrientationParameter(
                name="orientation"
            ),
            BuildingTemplateParameter(
                name="LightingPowerDensity",
                path="Loads.LightingPowerDensity",
                min=0.1,
                max=20
            ),
            BuildingTemplateParameter(
                name="EquipmentPowerDensity",
                path="Loads.EquipmentPowerDensity",
                min=0.1,
                max=20
            ),
            BuildingTemplateParameter(
                name="PeopleDensity",
                path="Loads.PeopleDensity",
                min=0.05,
                max=2,
            ),
            RValueParameter(
                name="FacadeRVaue",
                path="Facade",
                min=0.1,
                max=50,
            ),
            RValueParameter(
                name="RoofRVaue",
                path="Roof",
                min=0.1,
                max=50,
            ),
            RValueParameter(
                name="PartitionRValue",
                path="Partition",
                min=0.1,
                max=50,
            ),
            RValueParameter(
                name="SlabRVaue",
                path="Slab",
                min=0.1,
                max=50,
            ),
            SchemaParameter(
                name="schedules_seed",
                shape_ml=(0,),
                dtype="index",
            ),
            SchedulesParameters()
        ]
        self.storage_vec_len = 0
        self.ml_vec_len = 0
        self._key_ix_lookup = {}
        for i,parameter in enumerate(self.parameters):
            self._key_ix_lookup[parameter.name] = i
            parameter.start_storage = self.storage_vec_len
            parameter.start_ml = self.ml_vec_len
            self.storage_vec_len += parameter.len_storage
            self.ml_vec_len += parameter.len_ml
    
    @property
    def parameter_names(self):
        """Return a list of the named parameters in the schema"""
        return list(self.parameters.keys())

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
            desc += f"\nlocation storage: {parameter.start_storage}->{parameter.start_storage+parameter.len_storage} / location ml: {parameter.start_ml}->{parameter.start_ml+parameter.len_ml}"
            desc += f"\n"
        
        desc += f"\nTotal length of storage vectors: {self.storage_vec_len} / Total length of ml vectors: {self.ml_vec_len}"
        return desc
    
    def generate_empty_storage_vector(self):
        """
        Create a vector of zeros representing a blank storage vector
        Returns:
            storage_vector: np.ndarray, 1-dim 
        """
        return np.zeros(shape=self.storage_vec_len)

    def update_storage_vector(self, storage_vector, parameter, value):
        """
        Update a storage vector parameter with a value (or matrix which will be flattened)
        Args:
            storage_vector: np.ndarray, 1-dim, shape=(len(storage_vector))
            parameter: str, name of parameter to update
            value: np.ndarray | float, n-dim, will be flattened and stored in the storage vector
        """
        parameter = self[parameter]
        start = parameter.start_storage
        end = start + parameter.len_storage
        if isinstance(value, np.ndarray):
            storage_vector[start:end] = value.flatten()
        else:
            storage_vector[start] = value

    def update_storage_batch(self, storage_batch, index, parameter, value):
        """
        Update a storage vector parameter within a batch of storage vectors with a new value (or matrix which will be flattened)
        Args:
            storage_batch: np.ndarray, 2-dim, shape=(n_vectors, len(storage_vector))
            index: int | tuple, which storage vector (or range of storage vectors) within the batch to update.  use -1 if updating the full batch
            parameter: str, name of parameter to update
            value: np.ndarray | float, n-dim, will be flattened and stored in the storage vector
        """
        parameter = self[parameter]
        start = parameter.start_storage
        end = start + parameter.len_storage

        if isinstance(value, np.ndarray):
            value = value.reshape(-1,parameter.len_storage_vector)

        if isinstance(index, tuple):
            start_ix = index[0]
            end_ix = index[1]
            storage_batch[start_ix:end_ix, start:end] = value
        else:
            if index < 0:
                storage_batch[:, start:end] = value
            else:
                storage_batch[index, start:end] = value
        
class Model:
    __slots__ = (
        "design_vectors",
        "map"
    )


schema = Schema()
print(schema)