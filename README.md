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