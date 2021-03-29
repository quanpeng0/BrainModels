# BrainModels

[![LICENSE](https://anaconda.org/brainpy/brainpy/badges/license.svg)](https://github.com/PKU-NIP-Lab/BrainPy-Models)    [![Documentation](https://readthedocs.org/projects/brainpy/badge/?version=latest)](https://brainpy-models.readthedocs.io/en/latest/)     [![Conda](https://anaconda.org/brainpy/bpmodels/badges/version.svg)](https://anaconda.org/brainpy/bpmodels) 



``BrainModels`` provides standard and canonical brain models (including various neurons, synapses, networks, and intuitive paper examples) which are implemented with [BrainPy](https://brainpy.readthedocs.io/) simulator. Moreover, we welcome your brain model implementations, and publish them
through our [GitHub](https://github.com/PKU-NIP-Lab/BrainModels) homepage. In such a way, once your new model is implemented, it can be easily shared with other BrainPy users.

Currently, we provide the following models:


| Neuron models                                                | Synapse models                                               | Learning rules                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Leaky integrate-and-fire model](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.neurons.get_LIF.html) | [Alpha Synapse](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.synapses.get_alpha.html) | [STDP](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.learning_rules.get_STDP1.html) |
| [Hodgkin-Huxley model](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.neurons.get_HH.html) | [AMPA](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.synapses.get_AMPA1.html) / [NMDA](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.synapses.get_NMDA.html) | [BCM rule](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.learning_rules.get_BCM.html) |
| [Izhikevich model](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.neurons.get_Izhikevich.html) | [GABA_A](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.synapses.get_GABAa1.html) / [GABA_B](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.synapses.get_GABAb1.html) | [Oja\'s rule](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.learning_rules.get_Oja.html) |
| [Morris-Lecar model](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.neurons.get_MorrisLecar.html) | [Exponential Decay Synapse](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.synapses.get_exponential.html) |                                                              |
| [Generalized integrate-and-fire](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.neurons.get_GeneralizedIF.html) | [Difference of Two Exponentials](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.synapses.get_two_exponentials.html) |                                                              |
| [Exponential integrate-and-fire](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.neurons.get_ExpIF.html) | [Short-term plasticity](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.synapses.get_STP.html) |                                                              |
| [Quadratic integrate-and-fire](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.neurons.get_QuaIF.html) | [Gap junction](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.synapses.get_gap_junction.html) |                                                              |
| [adaptive Exponential IF](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.neurons.get_AdExIF.html) | [Voltage jump](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.synapses.get_voltage_jump.html) |                                                              |
| [adaptive Quadratic IF](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.neurons.get_AdQuaIF.html) |                                                              |                                                              |
| [Hindmarsh-Rose model](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.neurons.get_HindmarshRose.html) |                                                              |                                                              |
| [Wilson-Cowan model](https://brainpy-models.readthedocs.io/en/latest/apis/_autosummary/bpmodels.neurons.get_WilsonCowan.html) |                                                              |                                                              |



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


