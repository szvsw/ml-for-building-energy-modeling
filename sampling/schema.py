from typing import List
from functools import reduce

data_storage_parameters = [
    {
        "name": "id",
        "object": None,
        "form": "scalar",
        "type": "index",
        "ml_form": None
    },
    {
        "name": "base_template",
        "object": "template_lib",
        "form": "scalar",
        "type": "index",
        "ml_form": None
    },
    {
        "name": "base_epw",
        "object": "epw_file",
        "form": "scalar",
        "length": 1,
        "type": "index",
        "ml_form": None
    },
    {
        "name": "width",
        "object": "shoebox_zone_params",
        "form": "scalar",
        "type": "numeric",
        "config": {
            "min": 2,
            "max": 8
        },
        "ml_form": "scalar"
    },
    {
        "name": "facade_2_footprint",
        "object": "shoebox_zone_params",
        "form": "scalar",
        "type": "numeric",
        "config": {
            "min": 0.5,
            "max": 5
        },
        "ml_form": "scalar"
    },
    {
        "name": "perim_2_footprint",
        "object": "shoebox_zone_params",
        "form": "scalar",
        "type": "numeric",
        "config": {
            "min": 0.5,
            "max": 5
        },
        "ml_form": "scalar"
    },
    {
        "name": "roof_2_footprint",
        "object": "shoebox_zone_params",
        "form": "scalar",
        "type": "numeric",
        "config": {
            "min": 0.5,
            "max": 5
        },
        "ml_form": "scalar"
    },
    {
        "name": "footprint_2_ground",
        "object": "shoebox_zone_params",
        "form": "scalar",
        "type": "numeric",
        "config": {
            "min": 0.1,
            "max": 5
        },
        "ml_form": "scalar"
    },
    {
        "name": "shading_fact",
        "object": "shoebox_zone_params",
        "form": "scalar",
        "type": "fraction",
        "ml_form": "scalar"
    },
    {
        "name": "orientation",
        "object": "shoebox",
        "form": "scalar",
        "length": 4,
        "type": "discrete",
        "ml_form": "onehot"
    },
    {
        "name": "wwr",
        "object": "shoebox_wwr_map",
        "form": "vector",
        "length": 4,
        "type": "fraction",
        "ml_form": "vector"
    },
    {
        "name": "LightingPowerDensity",
        "object": "template",
        "form": "scalar",
        "type": "numeric",
        "config": {
            "path": ["Loads", "LightingPowerDensity"],
            "min": 0.1,
            "max": 20
        },
        "ml_form": "scalar"
    },
    {
        "name": "EquipmentPowerDensity",
        "object": "template",
        "form": "scalar",
        "type": "numeric",
        "config": {
            "path": ["Loads", "EquipmentPowerDensity"],
            "min": 0.1,
            "max": 20
        },
        "ml_form": "scalar"
    },
    {
        "name": "PeopleDensity",
        "object": "template",
        "form": "scalar",
        "type": "numeric",
        "config": {
            "path": ["Loads", "EquipmentPowerDensity"],
            "min": 0.1,
            "max": 2
        }
    },
    {
        "name": "DimmingType",
        "object": "template",
        "form": "scalar",
        "type": "discrete",
        "length": 3,
        "config": {
            "path": ["Loads", "DimmingType"],
            "min": 0.1,
            "max": 2,
            "default": 1
        },
        "ml_form": "onehot"
    },
    {
        "name": "IlluminanceTarget",
        "object": "template",
        "form": "scalar",
        "type": "numeric",
        "config": {
            "path": ["Loads", "IlluminanceTarget"],
            "min": 200,
            "max": 600,
        },
        "ml_form": "scalar"
    },
    {
        "name": "IsHeatingOn",
        "object": "template",
        "form": "scalar",
        "type": "boolean",
        "config": {
            "path": ["Conditioning", "IsHeatingOn"],
            "default": 1,
        },
        "ml_form": "scalar"
    },
    {
        "name": "IsCoolingOn",
        "object": "template",
        "form": "scalar",
        "type": "boolean",
        "config": {
            "path": ["Conditioning", "IsCoolingOn"],
            "default": 1,
        },
        "ml_form": "scalar"
    },
]

class WhiteboxMapParameter:
    __slots__ = (
        "name",
        "object",
        "form_storage",
        "form_ml",
        "type",
        "config",
        "start_storage",
        "start_ml",
        "reshape_storage",
        "shape_ml",
        "len_storage",
        "len_ml",
    )

    def __init__(self, name, shape_storage, shape_ml, **kwargs):
        self.name = name
        self.shape_storage = shape_storage
        self.shape_ml = shape_storage
        self.len_storage = reduce(lambda a,b: a*b, shape_storage)
        self.len_ml = reduce(lambda a,b: a*b, shape_ml)

    def get_control_values_from_storage_vect(self, storage_vector):
        pass

    def reshape_control_values(self, control_values):
        pass





class WhiteboxMap:
    __slots__ = (
        "parameters",
        "storage_vec_len",
        "ml_vec_len",
        "_key_ix_lookup"
    )

    def __init__(self, parameters: List[WhiteboxMapParameter]):
        self.parameters = parameters
        self.storage_vec_len = 0
        self.ml_vec_len = 0
        self._key_ix_lookup = {}
        for i,parameter in enumerate(self.parameters):
            self._key_ix_lookup[parameter.name] = i
            parameter.start_storage = self.storage_vec_len
            parameter.start_ml = self.ml_vec_len
            self.storage_vec_len += parameter.len_storage
            self.ml_vec_len += parameter.len_ml
