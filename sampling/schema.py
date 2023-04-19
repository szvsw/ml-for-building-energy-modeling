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
        "in_ml",
        "info"
    )

    def __init__(self, name, info, shape_storage=(1,), shape_ml=(1,), target_object_key=None, dtype="scalar"):
        self.name = name
        self.info = info
        self.target_object_key = target_object_key 
        self.dtype = dtype

        self.shape_storage = shape_storage
        self.shape_ml = shape_ml
        if shape_ml == (0,):
            self.in_ml = False

        self.len_storage = reduce(lambda a,b: a*b, shape_storage)
        self.len_ml = reduce(lambda a,b: a*b, shape_ml)
    
    def __repr__(self):
        return f"---{self.name}---\nshape_storage={self.shape_storage}, shape_ml={self.shape_ml}, dtype={self.dtype}\n{self.info}"

    def extract_storage_values(self, storage_vector):
        data = storage_vector[self.start_storage:self.start_storage+self.len_storage]
        if self.shape_storage == (1,):
            return data[0]
        else:
            return data.reshape(*self.shape_storage)
    
    def extract_storage_values_batch(self, storage_batch):
        data = storage_batch[:,self.start_storage:self.start_storage+self.len_storage]
        return data.reshape(-1,*self.shape_storage)
    
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
        super().__init__(name="schedules", dtype="matrix", shape_storage=(len(schedule_paths),len(operations)), shape_ml=(len(schedule_paths),8760), **kwargs)


    

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
                shape_ml=(0,),
                info="id of design"
            ),
            SchemaParameter(
                name="base_template",
                dtype="index",
                shape_ml=(0,),
                info="Lookup index of template to use."
            ),
            SchemaParameter(
                name="base_epw",
                dtype="index",
                shape_ml=(0,),
                info="Lookup index of EPW file to use."
            ),
            ShoeboxGeometryParameter(
                name="width",
                min=1.5,
                max=5,
                info="Width [m]"
            ),
            ShoeboxGeometryParameter(
                name="height",
                min=2.5,
                max=5,
                info="Height [m]",
            ),
            ShoeboxGeometryParameter(
                name="facade_2_footprint",
                min=0.5,
                max=5,
                info="Facade to footprint ratio (unitless)",
            ),
            ShoeboxGeometryParameter(
                name="perim_2_footprint",
                min=0.5,
                max=5,
                info="Perimeter to footprint ratio (unitless)",
            ),
            ShoeboxGeometryParameter(
                name="roof_2_footprint",
                min=0.5,
                max=5,
                info="Roof to footprint ratio (unitless)",
            ),
            ShoeboxGeometryParameter(
                name="footprint_2_ground",
                min=0.5,
                max=5,
                info="Footprint to ground ratio (unitless)",
            ),
            ShoeboxGeometryParameter(
                name="shading_fact",
                min=0,
                max=1,
                info="Shading fact (unitless)",
            ),
            ShoeboxGeometryParameter(
                name="wwr_n",
                min=0,
                max=1,
                info="Window-to-wall Ratio, N (unitless)",
            ),
            ShoeboxGeometryParameter(
                name="wwr_e",
                min=0,
                max=1,
                info="Window-to-wall Ratio, E (unitless)",
            ),
            ShoeboxGeometryParameter(
                name="wwr_s",
                min=0,
                max=1,
                info="Window-to-wall Ratio, S (unitless)",
            ),
            ShoeboxGeometryParameter(
                name="wwr_w",
                min=0,
                max=1,
                info="Window-to-wall Ratio, W (unitless)",
            ),
            ShoeboxOrientationParameter(
                name="orientation",
                info="Shoebox Orientation",
            ),
            BuildingTemplateParameter(
                name="LightingPowerDensity",
                path="Loads.LightingPowerDensity",
                min=0.1,
                max=20,
                info="Lighting Power Density [W/m2]",
            ),
            BuildingTemplateParameter(
                name="EquipmentPowerDensity",
                path="Loads.EquipmentPowerDensity",
                min=0.1,
                max=20,
                info="Equipment Power Density [W/m2]",
            ),
            BuildingTemplateParameter(
                name="PeopleDensity",
                path="Loads.PeopleDensity",
                min=0.05,
                max=2,
                info="People Density [people/m2]",
            ),
            RValueParameter(
                name="FacadeRValue",
                path="Facade",
                min=0.1,
                max=50,
                info="Facade R-value",
            ),
            RValueParameter(
                name="RoofRValue",
                path="Roof",
                min=0.1,
                max=50,
                info="Roof R-value",
            ),
            RValueParameter(
                name="PartitionRValue",
                path="Partition",
                min=0.1,
                max=50,
                info="Partition R-value",
            ),
            RValueParameter(
                name="SlabRValue",
                path="Slab",
                min=0.1,
                max=50,
                info="Slab R-value",
            ),
            SchemaParameter(
                name="schedules_seed",
                shape_ml=(0,),
                dtype="index",
                info="A seed to reliably reproduce schedules from the storage vector's schedule operations when generating ml vector",
            ),
            SchedulesParameters(
                info="A matrix in the storage vector with operations to apply to schedules; a matrix of timeseries in ml vector",
            )
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
            desc += f"\nlocation storage: {parameter.start_storage}->{parameter.start_storage+parameter.len_storage} / location ml: {parameter.start_ml}->{parameter.start_ml+parameter.len_ml}"
            desc += f"\n"
        
        desc += f"\nTotal length of storage vectors: {self.storage_vec_len} / Total length of ml vectors: {self.ml_vec_len}"
        return desc
    
    def generate_empty_storage_vector(self):
        """
        Create a vector of zeros representing a blank storage vector

        Returns:
            storage_vector: np.ndarray, 1-dim, shape=(len(storage_vector))
        """
        return np.zeros(shape=self.storage_vec_len)

    def generate_empty_storage_batch(self, n):
        """
        Create a matrix of zeros representing a batch of blank storage vectors

        Args:
            n: number of vectors to initialize in batch
        Returns:
            storage_batch: np.ndarray, 2-dim, shape=(n_vectors_in_batch, len(storage_vector))
        """
        return np.zeros(shape=(n,self.storage_vec_len))
    

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

    def update_storage_batch(self, storage_batch, index=None, parameter=None, value=None):
        """
        Update a storage vector parameter within a batch of storage vectors with a new value (or matrix which will be flattened)

        Args:
            storage_batch: np.ndarray, 2-dim, shape=(n_vectors, len(storage_vector))
            index: int | tuple, which storage vector (or range of storage vectors) within the batch to update.  omit or use None if updating the full batch
            parameter: str, name of parameter to update
            value: np.ndarray | float, n-dim, will be flattened and stored in the storage vector
        """
        parameter = self[parameter]
        start = parameter.start_storage
        end = start + parameter.len_storage

        if isinstance(value, np.ndarray):
            value = value.reshape(-1,parameter.len_storage)

        if isinstance(index, tuple):
            start_ix = index[0]
            end_ix = index[1]
            storage_batch[start_ix:end_ix, start:end] = value
        else:
            if index == None:
                storage_batch[:, start:end] = value
            else:
                storage_batch[index, start:end] = value
        
class Model:
    __slots__ = (
        "design_vectors",
        "map"
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
    values = np.random.rand(*shape) # create a random sample with appropriate shape
    schema.update_storage_batch(storage_batch, index=(start,end), parameter=parameter, value=values)
    print(schema[parameter].extract_storage_values_batch(storage_batch)[start-1:end+1]) # use [1:11] slice to see that the adjacentt cells are still zero

    """Updating an entire batch with random values"""
    parameter = "SlabRValue"
    n = batch_size
    shape = (n, *schema[parameter].shape_storage)
    values = np.random.rand(*shape) # create a random sample with appropriate shape
    schema.update_storage_batch(storage_batch, parameter=parameter, value=values)
    print(schema[parameter].extract_storage_values_batch(storage_batch))
