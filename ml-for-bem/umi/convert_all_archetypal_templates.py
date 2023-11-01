from mongoengine import connect
import os
import numpy as np
import pandas as pd
from typing import List
from mongoengine.queryset.visitor import Q
from umitemplatedb.mongodb_schema import BuildingTemplate
from dotenv import load_dotenv
from dotenv import dotenv_values
from tqdm import tqdm
from umi.ubem import Umi
import boto3

load_dotenv()
user = dotenv_values(".env").get("MONGODB_USER")
pw = dotenv_values(".env").get("MONGODB_PASSWORD")


server_addr = f"mongodb+srv://{user}:{pw}@ubemiocluster.cwe8w.mongodb.net/templates?retryWrites=true&w=majority"

# connect using mongoengine
connect(host=server_addr)

# count how many templates there are without fetching
print("Counting templates")
count = BuildingTemplate.objects.count()
print(f"There are {count} templates")
# get all the templates
print("Fetching templates")
page_size = 20
all_schedules = []
all_template_dicts = []
categories = [
    "Office",
    "School",
    "Residential",
    "Retail",
    "Hotel",
    "Market",
    "Mixed",
    "Laboratory",
    "MidRise",
    "Hospital",
    "Warehouse",
    "Restaurant",
    "Apartment",
    "SingleFamily",
    "MultiFamily",
]
for page_ix in tqdm(range(count // page_size + 1), desc="DB Page"):
    start_ix = page_ix * page_size
    end_ix = min((page_ix + 1) * page_size, count)
    templates: List[BuildingTemplate] = BuildingTemplate.objects[start_ix:end_ix]
    template_dicts = []
    for template in tqdm(templates, desc="Template"):
        new_temp = template.to_template()
        td = Umi.dict_from_buildingtemplate(building_template=new_temp)
        scheds = td.pop("schedules")
        all_schedules.append(scheds)
        td["TemplateName"] = template.Name
        td["ClimateZone"] = ", ".join(template.ClimateZone)
        if (
            template.Category is None
            or template.Category.lower() == "Uncategorized".lower()
            or template.Category == ""
        ):
            td["Category"] = ", ".join(
                cat for cat in categories if cat.lower() in template.Name.lower()
            )
        else:
            td["Category"] = ", ".join(template.Category)
        td["Country"] = ", ".join(template.Country)

        all_template_dicts.append(td)


df = pd.DataFrame.from_records(all_template_dicts)
df = df.reindex(range(len(df)))
all_schedules = np.array(all_schedules, dtype=np.float32)

client = boto3.client("s3")
bucket = "ml-for-bem"
experiment_name = "templates/v0"
df.to_hdf("data/ref_templates.hdf", key="features", mode="w")
np.save("data/ref_templates_schedules.npy", all_schedules)
client.upload_file(
    "data/ref_templates.hdf",
    bucket,
    f"{experiment_name}/ref_templates.hdf",
)
client.upload_file(
    "data/ref_templates_schedules.npy",
    bucket,
    f"{experiment_name}/ref_templates_schedules.npy",
)

os.remove("data/ref_templates.hdf")
os.remove("data/ref_templates_schedules.npy")

print(df.head())
