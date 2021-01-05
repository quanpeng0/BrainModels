.. BrainPy-Models documentation master file, created by
   sphinx-quickstart on Sat Oct 17 15:33:12 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BrainPy-Models documentation
===============================

``BrainPy-Models`` is based on `BrainPy <https://brainpy.readthedocs.io/>`_ neuronal dynamics simulation framework. Here you can find neurons, synapses models and topological networks implemented with BrainPy.

The prior goal of ``BrainPy-Models`` is to free users from repeatly implementing the most simple and commonly used models, instead they can import ``get_*()`` functions and take the advantage of our models.

.. note::

   We welcome your implementation about `neurons`, `synapses`, `learning rules`,
   `networks` and `paper examples`. https://github.com/PKU-NIP-Lab/BrainPy-Models



.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   
   tutorials/neurons
   tutorials/synapses

.. toctree::
   :maxdepth: 2
   :caption: Examples
   
   examples/neurons
   examples/synapses
   examples/learning_rules
   examples/networks
   examples/from_papers
   examples/dynamics_analysis

.. toctree::
   :maxdepth: 2
   :caption: API documentation
   
   apis/neurons
   apis/synapses
   apis/learning_rules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
