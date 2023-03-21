import time
import random
import os
import numpy as np
import pandas as pd

# from archetypal.schedule import Schedule
from archetypal import config, settings, parallel_process
from archetypal.idfclass.sql import Sql
from archetypal import UmiTemplateLibrary
from pyumi.shoeboxer.shoebox import ShoeBox

# config(use_cache=True, log_console=True)

epw_path = os.path.join('data', 'CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
template_path = os.path.join('data', 'BostonTemplateLibrary.json')
templates = UmiTemplateLibrary.open(template_path)
N_sims_in_benchmark = 100



def shoebox_from_template(
        name,
        template,
        zone_params=None,
        wwr=None,
        epw_path="C:/Users/zoele/Git_Repos/nested_grey_box/nested_grey_box/data/COMBINED_TRAINING.epw",
        path=None,
    ):
        """
        Builds idf from archetypal umi template and zone parameters
        zone_params: {
            "width": float (m)
            "height": float (m)
            "length": float (m)
            "facade_2_floor": float # facade area / total footprint area
            "core_2_perim": float # core area / perimeter area, core area = total area/perimeter area
            "roof_2_floor": float
            "footprint_2_floor": float
            "shading_fact": float (0-1)
        }
        """
        if zone_params is None:
            zone_params = {
                "width": 3.0+2*random.random(),
                "height": 3.0+random.random(),
                "facade_2_footprint": 0.3,
                "perim_2_footprint": 0.5,
                "roof_2_ground": 0.5,
                "footprint_2_ground": 0.5,
                "shading_fact": 0.8,
            }

        if wwr is None:
            wwr = random.random()*0.5+0.2

        wwr_map = {0: 0, 90: 0, 180: wwr, 270: 0}  # N is 0, E is 90
        # Convert to coords
        width = zone_params["width"]
        depth = zone_params["height"] / zone_params["facade_2_footprint"]
        perim_depth = depth * zone_params["perim_2_footprint"]
        zones_data = [
            {
                "name": "Perim",
                "coordinates": [
                    (width, 0),
                    (width, perim_depth),
                    (0, perim_depth),
                    (0, 0),
                ],
                "height": 3,
                "num_stories": 1,
                "zoning": "by_storey",
            },
            {
                "name": "Core",
                "coordinates": [
                    (width, perim_depth),
                    (width, depth),
                    (0, depth),
                    (0, perim_depth),
                ],
                "height": 3,
                "num_stories": 1,
                "zoning": "by_storey",
            },
        ]

        # if cls.verbose:
        #     print("Window to wall ratio assigned: ", wwr)
        #     print("Zone geometry assigned: ", zones_data)

        # zone_def = {
        #     "Core": template.Perimeter.to_dict(),
        #     "Perimeter": template.Perimeter.to_dict(),
        # }

        try:
            template.Perimeter.Loads.LightingPowerDensity = 10*random.random()
            # template.Core.Loads.EquipmentPowerDensity = 10*random.random()
            # template.Perimeter.Ventilation.InfiltrationAch = 3*random.random()
            sb = ShoeBox.from_template(
                building_template=template,
                zones_data=zones_data,
                wwr_map=wwr_map,
            )
            sb.epw = epw_path

            # Set floor and roof geometry for each zone
            for surface in sb.getsurfaces(surface_type="roof"):
                name = surface.Name
                name = name.replace("Roof", "Ceiling")
                # sb.add_adiabatic_to_surface(surface, name, zone_params["roof_2_ground"])
            for surface in sb.getsurfaces(surface_type="floor"):
                name = surface.Name
                name = name.replace("Floor", "Int Floor")
                # sb.add_adiabatic_to_surface(
                #     surface, name, zone_params["footprint_2_ground"]
                # )
            # Internal partition and glazing
            # Orientation

        except:
            raise
            # # TODO: add what the error is from eplus
            # sb = f"Error creating shoebox for {name}"

        # Ensure calculation of radiation on floor
        # Ensure 15 min time step

        return sb, zone_params

outputs_to_add = [
    dict(
        key="OUTPUT:VARIABLE",
        Variable_Name="Zone Ideal Loads Zone Total Heating Energy",
        Reporting_Frequency="hourly",
    ),
    dict(
        key="OUTPUT:VARIABLE",
        Variable_Name="Zone Ideal Loads Zone Total Cooling Energy",
        Reporting_Frequency="hourly",
    ),
    dict(
        key="OUTPUT:VARIABLE",
        Variable_Name="Lights Total Heating Energy",
        Reporting_Frequency="hourly",
    ),
    dict(
        key="OUTPUT:VARIABLE",
        Variable_Name="Zone Windows Total Transmitted Solar Radiation Energy",
        Reporting_Frequency="hourly",
    ),
]





rundict = {
    i: {"idx": i}
    for i in range(N_sims_in_benchmark)
}


def simulate(idx=0):
    sb, zone_params = shoebox_from_template(name=f"test_{idx:05d}", template=templates.BuildingTemplates[0], epw_path=epw_path)
    sb.outputs.add_custom(outputs_to_add)
    sb.outputs.apply()
    sb.simulate(verbose=False, prep_outputs=False, readvars=False)
    print("done simming")

start = time.time()
parallel_process(rundict, simulate, use_kwargs=True, processors=5)
end = time.time()

duration = end-start
it_time = duration/N_sims_in_benchmark
print(f"{N_sims_in_benchmark} sims took {duration:0.3f}s ({it_time}s/sim) [projected time for 1k sims: {int(1000*it_time/3600):02d}hrs]")