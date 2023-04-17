import os

import numpy as np

from functools import reduce

from archetypal import UmiTemplateLibrary
from archetypal.template import schedule

schedule_paths = [
    ["Loads", "EquipmentAvailabilitySchedule"],
    ["Loads", "LightsAvailabilitySchedule"],
    ["Loads", "OccupancySchedule"],
    ["Conditioning", "CoolingSchedule"],
    ["Conditioning", "HeatingSchedule"],
    # ["Conditioning", "HeatingSetpointSchedule"],
    # ["Conditioning", "HeatingSetpointSchedule"],
    ["Conditioning", "MechVentSchedule"],
    ["DomesticHotWater", "WaterSchedule"],
    # ["Ventilation", "NatVentSchedule"],
    ["Ventilation", "ScheduledVentilationSchedule"],
    # ["Windows", "ZoneMixingAvailabilitySchedule"],
    # ["Windows", "ShadingSystemAvailabilitySchedule"],
    # ["Windows", "AfnWindowAvailabilitySchedule"],
]

template_zones = ["Perimeter", "Core"]

"""
per bldg: seed
per sched: operations
"""

operations = [
    "roll","invert","scale","randomize","sine_0A","sine_0B","sine0C","sine_1A","sine_1B","sine_1C","synthetic","on","off" 
]

op_indices = {name: i for i,name in enumerate(operations)}

def get_schedules(template, zones=template_zones, paths=schedule_paths, operations = None):
    total_zones = len(zones)
    total_paths = len(paths)
    total_scheds = total_zones*total_paths
    scheds = np.zeros(shape=(total_scheds,8760))
    for i,zone in enumerate(zones):
        for j,path in enumerate(paths):
            scheds[i*total_paths + j] = get_sched_values(template, [zone]+path)
            if operations:
                pass
    return scheds

def get_sched_values(template, path):
    sched = reduce(lambda x, y: x[y], [template] + path)
    assert 'fraction' in sched.Type.Name.lower()
    return sched.all_values

def mutate_timeseries(series, operations):
    for i in range(series.shape[0]):
        """Handle rolling"""
        roll = operations[i, op_indices["roll"]]
        print(roll)
        series[i] = np.roll(series[i], int(roll))
    """Handle Inverting"""
    invert = operations[:, op_indices["invert"]]

    """Handle Scaling"""
    scalars = operations[:, op_indices["scale"]]
    series = series * scalars.reshape(-1,1)

    return series




if __name__ == "__main__":
    template_path = os.path.join('benchmark/data', 'BostonTemplateLibrary.json')
    templates = UmiTemplateLibrary.open(template_path)
    template = templates.BuildingTemplates[0]

    x = np.array([[1,2,3], [4,5,6]])
    print(mutate_timeseries(x, np.array([[0, 0, 1.5], [1, 0, 2]])))