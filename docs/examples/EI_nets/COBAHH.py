# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: brainmodels
#     language: python
#     name: brainmodels
# ---

# %% [markdown]
# # _(Brette, et, al., 2007)_ COBA-HH

# %% [markdown]
# Implementation of the paper:
#
# - Brette, R., Rudolph, M., Carnevale, T., Hines, M., Beeman, D., Bower, J. M., et al. (2007), Simulation of networks of spiking neurons: a review of tools and strategies., J. Comput. Neurosci., 23, 3, 349–98
#
# which is based on the balanced network proposed by:
#
# - Vogels, T. P. and Abbott, L. F. (2005), Signal propagation and logic gating in networks of integrate-and-fire neurons., J. Neurosci., 25, 46, 10786–95
#
#
# Authors:
#
# - Chaoming Wang (chao.brain@qq.com)

# %%



# %%
num_exc = 3200
num_inh = 800
Cm = 200  # Membrane Capacitance [pF]

gl = 10.  # Leak Conductance   [nS]
g_Na = 20. * 1000
g_Kd = 6. * 1000  # K Conductance      [nS]
El = -60.  # Resting Potential [mV]
ENa = 50.  # reversal potential (Sodium) [mV]
EK = -90.  # reversal potential (Potassium) [mV]
VT = -63.
V_th = -20.

# Time constants
taue = 5.  # Excitatory synaptic time constant [ms]
taui = 10.  # Inhibitory synaptic time constant [ms]

# Reversal potentials
Ee = 0.  # Excitatory reversal potential (mV)
Ei = -80.  # Inhibitory reversal potential (Potassium) [mV]

# excitatory synaptic weight
we = 6.  # excitatory synaptic conductance [nS]

# inhibitory synaptic weight
wi = 67.  # inhibitory synaptic conductance [nS]


import brainpy as bp
import brainmodels as bm
FixedProb = bp.connect.FixedProb

# channels => neurons
E = bp.CondNeuGroup(bm.Na.INa(), bm.K.IDR(), bm.other.IL())
I = bp.CondNeuGroup(bm.Na.INa(), bm.K.IDR(), bm.other.IL())
E.init(3200, monitors=['spike'])
I.init(800, monitors=['spike'])

# synapses
E2E = bm.ExpCOBA(E, E, FixedProb(prob=0.02), E=0, g_max=0.6, tau=5)
E2I = bm.ExpCOBA(E, I, FixedProb(prob=0.02), E=0, g_max=0.6, tau=5)
I2E = bm.ExpCOBA(I, E, FixedProb(prob=0.02), E=-80, g_max=6.7, tau=10)
I2I = bm.ExpCOBA(I, I, FixedProb(prob=0.02), E=-80, g_max=6.7, tau=10)

# neurons => network
net = bp.math.jit(bp.Network(E2E, E2I, I2I, I2E, E=E, I=I))
net.run(1000., report=0.2)
bp.visualize.raster_plot(E.mon.ts, E.mon.spike, show=True)

