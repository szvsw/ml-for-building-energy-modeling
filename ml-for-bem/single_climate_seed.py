import os
from utils.nrel_uitls import CLIMATEZONES, RESTYPES
import json
from pathlib import Path
import numpy as np
from archetypal import UmiTemplateLibrary
from schema import Schema
from shoeboxer.shoebox_config import ShoeboxConfiguration
from shoeboxer.builder import ShoeBox, template_dict

with open(Path(os.getcwd(), "ml-for-bem", "data", "city_map.json"), "r") as f:
    city_map = json.load(f)

schema = Schema()
storage_vector = schema.generate_empty_storage_vector()
for param in schema.parameters:
    try:
        val = param.mean
    except:
        val = 0
    schema.update_storage_vector(storage_vector, parameter=param.name, value=val)

cz_string = "4A"
cz_value = CLIMATEZONES[cz_string]
schema.update_storage_vector(
    storage_vector, parameter="climate_zone", value=cz_value
)  # Set all to be run in CZ4
schema.update_storage_vector(
    storage_vector, parameter="program_type", value=RESTYPES["Single-Family Detached"]
)
schema.update_storage_vector(
    storage_vector,
    parameter="base_epw",
    value=city_map["NY, New York"]["idx"],
)

shoebox_config = ShoeboxConfiguration()
shoebox_config.width = schema["width"].extract_storage_values(storage_vector)
shoebox_config.height = schema["height"].extract_storage_values(storage_vector)
shoebox_config.floor_2_facade = schema["floor_2_facade"].extract_storage_values(
    storage_vector
)
shoebox_config.core_2_perim = schema["core_2_perim"].extract_storage_values(
    storage_vector
)
shoebox_config.roof_2_footprint = schema["roof_2_footprint"].extract_storage_values(
    storage_vector
)
shoebox_config.ground_2_footprint = schema["ground_2_footprint"].extract_storage_values(
    storage_vector
)
shoebox_config.wwr = schema["wwr"].extract_storage_values(storage_vector)
shoebox_config.orientation = schema["orientation"].extract_storage_values(
    storage_vector
)
shoebox_config.shading_vect = np.zeros((12,))

# template_lib_idx = CLIMATEZONES["4A"]

# cz_value = int(cz_value)

# template_lib_path = Path(os.getcwd(), "ml-for-bem", "data", "template_libs", "cz_libs", "residential",f"CZ{cz_string}.json")

# n_masses = 2
# n_vintages = 4
# template_idx = (
#     n_masses * n_vintages * int(RESTYPES["Single-Family Detached"]) + n_masses * 2 + 0
# )

# lib = UmiTemplateLibrary.open(template_lib_path)
# template = lib.BuildingTemplates[template_idx]
# schedules = schema.parameters[-1].extract_from_template(template)
# # Save schedules as npy file
# np.save(Path(os.getcwd(), "ml-for-bem", "data", "residential_schedules.npy"), schedules)

schedules = np.load(
    Path(os.getcwd(), "ml-for-bem", "data", "residential_schedules.npy")
)

sb = ShoeBox.from_vector(
    name="shoebox_test",
    schema=schema,
    shoebox_config=shoebox_config,
    vector=storage_vector,
    schedules=schedules,
    change_summary=False,
    output_directory=Path(os.getcwd(), "cache"),
)
idf = sb.idf(run_simulation=True)
idf.view_model()
