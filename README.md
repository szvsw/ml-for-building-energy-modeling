# Machine Learning for Building Energy Modeling

MIT 4.463 class project page, Zoe LeHong & Sam Wolk

## Sampling Setup

First you will need to clone and checkout the desired branches of [Archetypal](https://github.com/samuelduchesne/archetypal) and [PyUmi](https://github.com/samuelduchesne/pyumi).

Then, create a new Conda env using the provided yml file.

- ```
conda env create -n ml-for-bem-sampling -f environment_sampling.yml
conda activate ml-for-bem-sampling
cd /path/to/pyumi
pip install -r requirements.txt
pip install ladybug-core
pip install -e . --no-deps
cd /path/to/archetypal
pip install -e .
```


## Sampling Outline

1. 500 building baselines from Res(Com)Stock --> identify baseline building template that following mutations apply to.
1. 2k after Grid Sample 4 Orientations
1. 20k after LHC on geometric/internal mass etc (10)
1. 200k after Randomly perturb schedules 10 times (including on/off consts) 
1. 1000k after applying upgrades/downgrades




ramps
impulses
shape=(500,4,10,10,-1[5])

setpoint occupancy lighting equipment [dhw] []
building 1a ...  1  1  1  1 1 0   ...
building 1b ... -1 -1 -1 -1 -2 0 
building 1b ...  0 0 0 0 0 0 
building 1b ...  0 [0-12] 
.
.
building 1j