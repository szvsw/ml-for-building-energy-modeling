import os

import numpy as np
import matplotlib.pyplot as plt

from functools import reduce

from archetypal import UmiTemplateLibrary
from archetypal.template.schedule import UmiSchedule

schedule_paths = [
    ["Loads", "EquipmentAvailabilitySchedule"],
    ["Loads", "LightsAvailabilitySchedule"],
    ["Loads", "OccupancySchedule"],
    # ["Conditioning", "MechVentSchedule"],
    # ["Conditioning", "CoolingSchedule"],
    # ["Conditioning", "HeatingSchedule"],
    # ["Conditioning", "HeatingSetpointSchedule"],
    # ["Conditioning", "HeatingSetpointSchedule"],
    # ["DomesticHotWater", "WaterSchedule"],
    # ["Ventilation", "NatVentSchedule"],
    # ["Ventilation", "ScheduledVentilationSchedule"],
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
    "on/off",
    "uniform_random_daily",
    "pulse_period",
    "pulse_width",
]

op_indices = {name: i for i, name in enumerate(operations)}


def get_schedules(template, zones=template_zones, paths=schedule_paths):
    """
    Takes in a template, which zones to retrieve, which schedules within those zones, and fetches the schedules
    Args:
        template: archetypal building template
        zones: list(str) e.g. ["Perimeter"] or ["Perimeter, Core"]
        paths: list(list(str)): which paths to fetch from each zone, e.g. [ ["Loads", "EquipmentAvailabilitySchedule"], ["Conditioning", "CoolingSchedule"]]
    returns:
        scheds: (n_zones * n_paths, 8760)
    """
    total_zones = len(zones)
    total_paths = len(paths)
    total_scheds = total_zones * total_paths
    scheds = np.zeros(shape=(total_scheds, 8760))
    for i, zone in enumerate(zones):
        for j, path in enumerate(paths):
            scheds[i * total_paths + j] = get_sched_values(template, [zone] + path)
    return scheds


def get_sched_values(template, path):
    """
    Takes in a building template and a path and returns the schedule at that path.
    Allows for dynamically fetching schedules programmatically

    Args:
        template: an archetypal building template
        path: list(str), e.g. ["Perimeter", "Loads", "EquipmentAvailabilitySchedule"]
    Returns:
        schedule: an archetypal UmiSchedule
    """
    sched = reduce(lambda x, y: x[y], [template] + path)
    assert "fraction" in sched.Type.Name.lower()
    return sched.all_values


def mutate_timeseries(series, operations, seed):
    """
    Takes in a matrix of time series and a matrix of operations to
    apply to the time series, along with a seed for reproducible randomization

    Args:
        series:     (n_schedules, 8760) a matrix of time series, where each row is a different schedule
        operations: (n_schedules, ),    each schedule has a vector which controls which operations are applied (and how)
        seed:       int,                make random operations reproducible
    Returns:
        series:     (n_schedules, 8760) mutated time series matrix
    """
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
    t_cycle = np.linspace(0, 2 * np.pi, n)
    for i in range(n_series):
        rev = operations[i, op_indices["reverse"]]
        roll = operations[i, op_indices["roll"]]
        invert = operations[i, op_indices["invert"]]
        scale = operations[i, op_indices["scale"]]
        bias = operations[i, op_indices["bias"]]
        noise = operations[i, op_indices["noise"]]

        sin_overwrite = operations[i, op_indices["sin_overwrite"]]
        sin_bias = operations[i, op_indices["sin_bias"]]
        sin_0_amp = operations[i, op_indices["sin_0_amp"]]
        sin_0_freq = operations[i, op_indices["sin_0_freq"]]
        sin_0_phase = operations[i, op_indices["sin_0_phase"]]
        sin_1_amp = operations[i, op_indices["sin_1_amp"]]
        sin_1_freq = operations[i, op_indices["sin_1_freq"]]
        sin_1_phase = operations[i, op_indices["sin_0_phase"]]

        # TODO: Unimplemented
        synthetic = operations[i, op_indices["synthetic"]]

        on_off = operations[i, op_indices["on/off"]]
        uniform_random_daily = [i, op_indices["uniform_random_daily"]]
        pulse_period = operations[i, op_indices["pulse_period"]]
        pulse_width = [i, op_indices["pulse_width"]]

        """Handle Reversing"""
        if rev == 1:
            series[i] = np.flip(series[i])

        """Handle rolling"""
        series[i] = np.roll(series[i], int(roll))

        """Handle Inverting"""
        if invert == 1:
            series[i] = 1 - series[i]

        """Handle Scaling"""
        series[i] *= scale
        series[i] = np.clip(series[i], 0, 1)

        """Handle Biasing"""
        series[i] += bias
        series[i] = np.clip(series[i], 0, 1)

        """Handle Noise"""
        series[i] += np.random.rand(n) * noise
        series[i] = np.clip(series[i], 0, 1)

        """Handle Sine"""
        if sin_overwrite == 1:
            series[i] = 0

        series[i] += (
            sin_0_amp * (0.5 * np.sin(sin_0_freq * (t_cycle + sin_0_phase)) + 0.5)
            + sin_1_amp * (0.5 * np.sin(sin_1_freq * (t_cycle + sin_1_phase)) + 0.5)
            + sin_bias
        )
        series[i] = np.clip(series[i], 0, 1)

        """Handle Consts"""
        if on_off == 1:
            series[i] = np.ones(n)
        elif on_off == -1:
            series[i] = np.zeros(n)

        """Handle Uniform Random Daily"""
        if uniform_random_daily == 1:
            week = np.rand(7 * 24)
            year = np.tile(week, 55)
            year = year[:8760]
            series[i] = year

        """Handle Pulse"""
        if pulse_period > 0:
            cycle = np.rand(pulse_period)
            cycle = np.where(cycle > 0.5, 0 * cycle + 1, 0 * cycle)
            year = np.tile(cycle, int(8760 / pulse_period) + 2)
            year = year[:8760]
            series[i] = year

    return series


def update_schedule_objects(
    template, timeseries, zones=template_zones, paths=schedule_paths, id=0
):
    """
    This method takes in a template and a matrix of timeseries along with a map for where the the timeseries apply to.
    It then creates new schedules and overwrites the old ones with the appropriate schedules.
    This is to prevent shared schedules from creating a (deterministic) race condition.

    Args:
        template (archetypal BuildingTemplate): The building template to udpate
        timeseries (np.ndarray): A matrix of timeseries to use (shape=(n_schedules,8760))
        zones (List[str]): a list of zones to update
        paths (List[List[str]]): a list of paths within those zones
        id (int | str): an id to add to the unique names of the new schedules
    Returns:
        None
    """
    total_zones = len(zones)
    total_paths = len(paths)
    total_scheds = total_zones * total_paths
    if total_scheds == 1:
        assert timeseries.shape == (8760)
    else:
        assert timeseries.shape == (total_scheds, 8760)
    for i, zone in enumerate(zones):
        for j, path in enumerate(paths):
            schedule_array = timeseries[i * total_paths + j]
            name = ".".join([str(id), zone, *path])
            new_schedule = UmiSchedule.from_values(
                Name=name, Values=schedule_array
            ).to_year_week_day()[0]
            target_schedule_attr = path[-1]
            parent_path = [template, zone, *path]
            parent_path = parent_path[:-1]
            parent = reduce(lambda a, b: a[b], parent_path)
            setattr(parent, target_schedule_attr, new_schedule)


def extract_schedules_from_flattened_vectors(vecs, start, n_schedules):
    """
    Given a matrix of flattened design vectors, extract the timeseries data as a channel indexed tensor
    Args:
        vecs: (n_design_vectors, n_design_parameters_per_vector) - the 2D matrix of flattened design vectors
        start: int - where in the flattened design vector the schedules start at
        ct: the number of schedules to extract
    Returns:
        scheds: (n_design_vectors, n_schedules, 8760) - a tensor with the schedules extracted
    """
    # TODO: change to torch
    scheds = vecs[:, start : start + n_schedules * 8760]
    return scheds.reshape(-1, n_schedules, 8760)


if __name__ == "__main__":

    # Open a template lib
    template_path = os.path.join("benchmark/data", "BostonTemplateLibrary.json")
    templates = UmiTemplateLibrary.open(template_path)

    # Pick a template for testing
    template = templates.BuildingTemplates[0]
    template_b = templates.BuildingTemplates[1]

    # Zones to use in testing
    zones = ["Perimeter"]

    # Paths to get in chosen zones
    paths = [
        ["Loads", "EquipmentAvailabilitySchedule"],
        ["Loads", "LightsAvailabilitySchedule"],
    ]

    # Fetch schedules
    scheds = get_schedules(template, zones=zones, paths=paths)
    scheds_b = get_schedules(
        template, zones=zones, paths=paths
    )  # get an alternate copy for testing some of the tensor ops

    # set a seed - this will be stored in the building's design vector
    seed = 42

    # mutate the time series
    res = mutate_timeseries(
        series=scheds,
        operations=np.array(
            [
                [0, 0, 0, 1.0, 0.0, 0.00, 1, 0, 0.5, 1, 0, 0.25, 365, 0, 0, 0],
                [0, 0, 0, 0.0, 0.0, 1.00, 0, 0, 0.0, 0, 0, 0.0, 0.0, 0, 0, 0],
            ]
        ),
        seed=seed,
    )

    # plut the mutations
    for i in range(scheds.shape[0]):
        figs, axs = plt.subplots(1, 2)
        axs[0].plot(scheds[i, : 7 * 24])
        plt.title("Original/Mutated")
        axs[1].plot(res[i, : 7 * 24])
    plt.show()

    # create some dummy vectors
    dummy_start = np.zeros(100)
    dummy_end = np.zeros(100)
    # stack data into an array of flattened design vectors to replicate batch training
    # shape = (n_design_vectors, n_parameters_per_vector)
    design_vectors = np.stack(
        [
            np.concatenate([dummy_start, scheds.flatten(), dummy_end]),
            np.concatenate([dummy_start, scheds_b.flatten(), dummy_end]),
        ]
    )
    print(design_vectors.shape)
    # extract schedules from design vectors as tensor, similar to a multi-channel image, but in 1D (pytorch expects shape=(n_samples, channels, spatial_dimension))
    # shape = (n_design_vectors, n_schedules, 8760)
    schedule_tensor = extract_schedules_from_flattened_vectors(
        design_vectors, start=100, n_schedules=2
    )
    print(schedule_tensor.shape)
