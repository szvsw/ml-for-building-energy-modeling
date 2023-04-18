import os

import numpy as np
import matplotlib.pyplot as plt

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
    "reverse",
    "roll",
    "invert",
    "scale",
    "bias",
    "noise",
    "sin_overwrite",
    "sin_bias",
    "sin_0_amp",
    "sin_0_freq",
    "sin_0_phase",
    "sin_1_amp",
    "sin_1_freq",
    "sin_1_phase",
    "synthetic",
    "on/off"
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

def mutate_timeseries(series, operations, seed):
    series = series.copy()
    np.random.seed(seed)
    n = 0
    n_series = 0
    if len(series.shape) > 1:
        n = series.shape[1]
        n_series = series.shape[0]
    else:
        n = series.shape[0]
        n_series = 1
    t_cycle = np.linspace(0, 2*np.pi, n)
    for i in range(n_series):
        rev           = operations[i, op_indices["reverse"]]
        roll          = operations[i, op_indices["roll"]]
        invert        = operations[i, op_indices["invert"]]
        scale         = operations[i, op_indices["scale"]]
        bias          = operations[i, op_indices["bias"]]
        noise         = operations[i, op_indices["noise"]]

        sin_overwrite = operations[i, op_indices["sin_overwrite"]]
        sin_bias      = operations[i, op_indices["sin_bias"]]
        sin_0_amp     = operations[i, op_indices["sin_0_amp"]]
        sin_0_freq    = operations[i, op_indices["sin_0_freq"]]
        sin_0_phase   = operations[i, op_indices["sin_0_phase"]]
        sin_1_amp     = operations[i, op_indices["sin_1_amp"]]
        sin_1_freq    = operations[i, op_indices["sin_1_freq"]]
        sin_1_phase   = operations[i, op_indices["sin_0_phase"]]

        # TODO: Unimplemented
        synthetic = operations[i, op_indices["synthetic"]]

        on_off = operations[i, op_indices["on/off"]]

        """Handle Reversing"""
        if rev == 1:
            series[i] = np.flip(series[i])

        """Handle rolling"""
        series[i] = np.roll(series[i], int(roll))

        """Handle Inverting"""
        if invert == 1:
            series[i] = 1-series[i]
        
        """Handle Scaling"""
        series[i] *= scale
        series[i] = np.clip(series[i], 0, 1)

        """Handle Biasing"""
        series[i] += bias
        series[i] = np.clip(series[i], 0, 1)


        """Handle Noise"""
        series[i] += np.random.rand(n)*noise
        series[i] = np.clip(series[i], 0, 1)

        """Handle Sine"""
        if sin_overwrite == 1:
            series[i] = 0

        series[i] += sin_0_amp * (0.5*np.sin(sin_0_freq*(t_cycle + sin_0_phase))+0.5) + sin_1_amp * (0.5*np.sin(sin_1_freq*(t_cycle + sin_1_phase))+0.5) + sin_bias
        series[i] = np.clip(series[i], 0, 1)

        """Handle Consts"""
        if on_off == 1:
            series[i] = np.ones(n)
        elif on_off == -1:
            series[i] = np.zeros(n)


    return series




if __name__ == "__main__":
    template_path = os.path.join('benchmark/data', 'BostonTemplateLibrary.json')
    templates = UmiTemplateLibrary.open(template_path)
    template = templates.BuildingTemplates[0]

    zones = ["Perimeter"]
    paths = [
        ["Loads", "EquipmentAvailabilitySchedule"],
        ["Loads", "LightsAvailabilitySchedule"]
    ]
    scheds = get_schedules(template,zones=zones, paths=paths)
    print(scheds.shape)
    # np.random.seed(1)
    # x = np.random.rand(2,8760)
    seed = 42
    res = mutate_timeseries(
        scheds, 
        np.array([
            [0, 0, 0, 1.0, 0.0, 0.00, 1, 0, 0.5, 1, 0, 0.25, 365, 0, 0, 0], 
            [0, 0, 0, 0.0, 0.0, 1.00, 0, 0, 0.0, 0, 0, 0.0, 0.0, 0, 0, 0], 
        ]),
        seed
    )
    for i in range(scheds.shape[0]):
        figs, axs = plt.subplots(1,2)
        axs[0].plot(scheds[i, :7*24])
        plt.title("Original/Mutated")
        axs[1].plot(res[i, :7*24])
    plt.show()