import numpy as np
import os
from pathlib import Path

from utils.constants import DEFAULT_SCHEDULES_PATH

module_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

SCHEDULES_ARR_PATH = Path(module_path, DEFAULT_SCHEDULES_PATH)

default_schedules = np.load(SCHEDULES_ARR_PATH)


def schedules_from_seed(seed: int):
    # step 1: load the seedo
    np.random.seed(int(seed))

    # step 2: decide whether or not to use random or schedules
    is_fully_rand = np.random.choice([True, False], size=1, p=[0.8, 0.2])

    # step 3: if is fully_rand, then use random, else use schedules
    if is_fully_rand:
        scheds = np.random.rand(3, 8760)
    else:
        # Step 3a. select which archetype to use
        sched_type = np.random.choice(range(default_schedules.shape[0]))
        # Step 3b. select schedules
        scheds = default_schedules[sched_type]
        # Step 3c. mutate schedules with nosie
        noise_amp = np.random.rand(3) * 0.25
        noise = np.random.rand(3, 8760) * noise_amp.reshape(3, 1)
        scheds = scheds + noise
        # step 3d. clip to 0-1
        scheds = np.clip(scheds, 0, 1)

    return scheds
