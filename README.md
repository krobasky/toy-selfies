# toy-selfies

This repository demonstrates modelling over SELFIES strings. According to selfies.readthedocs.io:

> SELFIES (SELF-referencIng Embedded Strings) is a 100% robust
  molecular string representation. A main objective is to use SELFIES
  as direct input into machine learning models, in particular in
  generative models, for the generation of outputs with guaranteed
  validity.

## Manifest

```
src/vae.ipynb: jupyter notebook demonstrating SELFIES generation from simple, pytorch-implemented VAE model
src/config: full SELFIES vocabulary for the QM9 dataset from the “MoleculeNet: A Benchmark for Molecular Machine Learning” paper, consisting of about 130,000 molecules.
src/bd: source code supporting the notebook
src/bd/data.py: QM9 data loader and utilities
src/bd/models.py: VAE model pytorch source code
src/bd/asses.py: functions for assessing the model
src/bd/log.py: simple logging methods to support clearn code
```

## Usage

See the `src/vae.ipynb` notebook for usage.

