import os
from typing import List
from pathlib import Path
import functools
import operator

import json
from dotenv import dotenv_values, load_dotenv
from mongoengine import connect

from mongoengine.queryset.visitor import Q
from tqdm import tqdm
import umitemplatedb.mongodb_schema as utdb

from archetypal import UmiTemplateLibrary

import pandas as pd

load_dotenv()
user = dotenv_values(".env").get("MONGODB_USER")
pw = dotenv_values(".env").get("MONGODB_PASSWORD")

server_addr = f"mongodb+srv://{user}:{pw}@ubemiocluster.cwe8w.mongodb.net/templates?retryWrites=true&w=majority"

# connect using mongoengine
connect(host=server_addr)

windows = {}
# czs = ["1A", "4C", "6B"]
czs = ["4C", "6B"]
for c in czs:
    qsets = [Q(ClimateZone=c)]
    filters = functools.reduce(operator.ior, qsets)
    # count how many templates there are without fetching
    print("Counting templates")
    count = utdb.BuildingTemplate.objects(filters).count()
    print(f"There are {count} templates")
    # get all the templates
    print("Fetching templates")
    page_size = 20
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

    output_dir = "D:/Users/zoelh/Dropbox (MIT)/Thesis Work/Building Energy References/ArchetypalReferences"

    building_template_list = []
    for page_ix in tqdm(range(count // page_size + 1), desc="DB Page"):
        start_ix = page_ix * page_size
        end_ix = min((page_ix + 1) * page_size, count)
        templates: List[utdb.BuildingTemplate] = utdb.BuildingTemplate.objects(filters)[
            start_ix:end_ix
        ]

        for template in tqdm(templates, desc="Template"):
            try:
                name = template.Name.split("_")[0] + "_" + c
                pth = Path(output_dir, f"{name}.json")
                if not os.path.isfile(pth):
                    new_temp = template.to_template()
                    print(template.Name)
                    print("U-value", new_temp.Windows.Construction.u_value)
                    print(
                        "Tvis",
                        new_temp.Windows.Construction.Layers[
                            0
                        ].Material.VisibleTransmittance,
                    )
                    if len(new_temp.Windows.Construction.Layers) > 1:
                        print(
                            "SHGC",
                            new_temp.Windows.Construction.Windows.Construction.shgc(),
                        )
                        shgc = new_temp.Windows.Construction.Windows.Construction.shgc()
                    else:
                        print("Single pane, shgc")
                        shgc = None
                    windows[template.Name] = {
                        "u_value": new_temp.Windows.Construction.u_value,
                        "shgc": shgc,
                        "t_vis": new_temp.Windows.Construction.Layers[
                            0
                        ].Material.VisibleTransmittance,
                    }
                    building_template_list.append(new_temp)
                    template_lib = UmiTemplateLibrary(
                        BuildingTemplates=building_template_list
                    )
                    template_lib.unique_components()  # Update to unique components list
                    # TODO: sometimes gas materials aren't deduped when the original bts are unpacked
                    template_lib.unique_components("GasMaterials")
                    with open(pth, "w") as f:
                        json.dump(template_lib.to_dict(), f, indent=4)
            except Exception as e:
                print("TEMPLATE ERROR: ", e)

    windows_df = pd.DataFrame.from_dict(windows)
    windows_df.T.to_csv(Path(output_dir, f"DefaultTemplateLibraryWindows_{c}.csv"))

    # template_lib = UmiTemplateLibrary(BuildingTemplates=building_template_list)
    # template_lib.unique_components()  # Update to unique components list
    # # TODO: sometimes gas materials aren't deduped when the original bts are unpacked
    # template_lib.unique_components("GasMaterials")

    # with open(Path(output_dir, f"DefaultTemplateLibrary_{c}.json"), "w") as f:
    #     json.dump(template_lib.to_dict(), f, indent=4)
