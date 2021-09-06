# BrainModels

[![LICENSE](https://img.shields.io/github/license/PKU-NIP-Lab/BrainModels)](https://github.com/PKU-NIP-Lab/BrainPy-Models)    [![Documentation](https://readthedocs.org/projects/brainpy/badge/?version=latest)](https://brainmodels.readthedocs.io/en/latest/)    [![PyPI version](https://badge.fury.io/py/brainmodels.svg)](https://badge.fury.io/py/brainmodels)

``BrainModels`` provides standard and canonical brain models (including various neurons, synapses, networks, and intuitive paper examples) which are implemented with [BrainPy](https://brainpy.readthedocs.io/) simulator. Moreover, we welcome your brain model implementations, and publish them through our [GitHub](https://github.com/PKU-NIP-Lab/BrainModels) homepage. In such a way, once your new model is implemented, it can be easily shared with other BrainPy users.



Currently, we provide the following standard models:


| Neuron Models                        | Synapse Models                 | Learning Rules |
| ------------------------------------ | ------------------------------ | -------------- |
| Leaky integrate-and-fire model       | Alpha Synapse                  | STDP           |
| Exponential integrate-and-fire model | AMPA / NMDA                    | BCM rule       |
| Quadratic integrate-and-fire         | GABAA / GABAB                  | Oja\'s rule    |
| Adaptive Exponential IF model        | Exponential Decay Synapse      |                |
| Adaptive Quadratic IF model          | Difference of Two Exponentials |                |
| Generalized IF model                 | Short-term plasticity          |                |
| Izhikevich model                     | Gap junction                   |                |
| Hodgkin-Huxley model                 | Voltage jump                   |                |
| Morris-Lecar model                   |                                |                |
| Hindmarsh-Rose model                 |                                |                |



## Installation

Install `BrainModels` using `pip`:

```bash
> pip install brainmodels
```

Install from source code:

```bash
> pip install git+https://github.com/PKU-NIP-Lab/BrainModels
```



`BrainModels` is based on Python (>=3.6), and the following packages need to be installed to use `BrainModels`:

-   brain-py >= 1.1.0

