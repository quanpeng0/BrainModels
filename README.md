# BrainModels

[![LICENSE](https://anaconda.org/brainpy/brainpy/badges/license.svg)](https://github.com/PKU-NIP-Lab/BrainPy-Models)    [![Documentation](https://readthedocs.org/projects/brainpy/badge/?version=latest)](https://brainmodels.readthedocs.io/en/latest/)     [![Conda](https://anaconda.org/brainpy/bpmodels/badges/version.svg)](https://anaconda.org/brainpy/bpmodels) 



``BrainModels`` provides standard and canonical brain models (including various neurons, synapses, networks, and intuitive paper examples) which are implemented with [BrainPy](https://brainpy.readthedocs.io/) simulator. Moreover, we welcome your brain model implementations, and publish them
through our [GitHub](https://github.com/PKU-NIP-Lab/BrainModels) homepage. In such a way, once your new model is implemented, it can be easily shared with other BrainPy users.

Currently, we provide the following models:


| Neuron models                  | Synapse models                 | Learning rules |
| ------------------------------ | ------------------------------ | -------------- |
| Leaky integrate-and-fire model | Alpha Synapse                  | STDP           |
| Hodgkin-Huxley model           | AMPA / NMDA                    | BCM rule       |
| Izhikevich model               | GABA_A / GABA_B                | Oja\'s rule    |
| Morris-Lecar model             | Exponential Decay Synapse      |                |
| Generalized integrate-and-fire | Difference of Two Exponentials |                |
| Exponential integrate-and-fire | Short-term plasticity          |                |
| Quadratic integrate-and-fire   | Gap junction                   |                |
| adaptive Exponential IF        | Voltage jump                   |                |
| adaptive Quadratic IF          |                                |                |
| Hindmarsh-Rose model           |                                |                |
| Wilson-Cowan model             |                                |                |



# Installation

Install `BrainModels` using `pip`:

```bash
> pip install brainmodels
```

Install ``BrainModels`` using ``conda``:

```bash
> conda install brainmodels -c brainpy 
```

Install from source code:

```bash
> pip install git+https://github.com/PKU-NIP-Lab/BrainModels
> # or
> git clone https://github.com/PKU-NIP-Lab/BrainModels
> cd BrainModels
> python setup.py install
```

``BrainModels`` is based on Python (>=3.7), and the following packages need to be installed to use `BrainModels`:

-   brainpy-simulator >= 1.0.0
-   Matplotlib >= 3.2
-   NumPy >= 1.13

