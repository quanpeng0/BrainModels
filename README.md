# BrainModels

[![LICENSE](https://anaconda.org/brainpy/brainpy/badges/license.svg)](https://github.com/PKU-NIP-Lab/BrainPy-Models)    [![Documentation](https://readthedocs.org/projects/brainpy/badge/?version=latest)](https://brainmodels.readthedocs.io/en/latest/)     [![Conda](https://anaconda.org/brainpy/bpmodels/badges/version.svg)](https://anaconda.org/brainpy/bpmodels) 



``BrainModels`` provides standard and canonical brain models (including various neurons, synapses, networks, and intuitive paper examples) which are implemented with [BrainPy](https://brainpy.readthedocs.io/) simulator. Moreover, we welcome your brain model implementations, and publish them
through our [GitHub](https://github.com/PKU-NIP-Lab/BrainModels) homepage. In such a way, once your new model is implemented, it can be easily shared with other BrainPy users.

Currently, we provide the following models:


| Neuron models                                                | Synapse models                                               | Learning rules                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Leaky integrate-and-fire model](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.neurons.LIF.html) | [Alpha Synapse](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.synapses.Alpha.html) | [STDP](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.learning_rules.STDP1.html) |
| [Hodgkin-Huxley model](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.neurons.HH.html) | [AMPA](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.synapses.AMPA1.html) / [NMDA](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.synapses.NMDA.html) | [BCM rule](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.learning_rules.BCM.html) |
| [Izhikevich model](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.neurons.Izhikevich.html) | [GABA_A](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.synapses.GABAa1.html) / [GABA_B](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.synapses.GABAb1.html) | [Oja\'s rule](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.learning_rules.Oja.html) |
| [Morris-Lecar model](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.neurons.MorrisLecar.html) | [Exponential Decay Synapse](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.synapses.Exponential.html) |                                                              |
| [Generalized integrate-and-fire](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.neurons.GeneralizedIF.html) | [Difference of Two Exponentials](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.synapses.Two_exponentials.html) |                                                              |
| [Exponential integrate-and-fire](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.neurons.ExpIF.html) | [Short-term plasticity](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.synapses.STP.html) |                                                              |
| [Quadratic integrate-and-fire](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.neurons.QuaIF.html) | [Gap junction](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.synapses.Gap_junction.html) |                                                              |
| [adaptive Exponential IF](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.neurons.AdExIF.html) | [Voltage jump](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.synapses.Voltage_jump.html) |                                                              |
| [adaptive Quadratic IF](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.neurons.AdQuaIF.html) |                                                              |                                                              |
| [Hindmarsh-Rose model](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.neurons.HindmarshRose.html) |                                                              |                                                              |
| [Wilson-Cowan model](https://brainmodels.readthedocs.io/en/latest/apis/_autosummary/brainmodels.tensor_backend.neurons.WilsonCowan.html) |                                                              |                                                              |



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

