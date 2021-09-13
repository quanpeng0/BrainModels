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
# # Example of Hindmarsh-Rose model

# %% [markdown]
# Reference: 
#
# - Hindmarsh, James L., and R. M. Rose. "*A model of neuronal bursting using three coupled first order differential equations.*" Proceedings of the Royal society of London. Series B. Biological sciences 221.1222 (1984): 87-102.
# - Storace, Marco, Daniele Linaro, and Enno de Lange. "*The Hindmarsh–Rose neuron model: bifurcation analysis and piecewise-linear approximations.*" Chaos: An Interdisciplinary Journal of Nonlinear Science 18.3 (2008): 033128.

# %% [markdown]
# Hindmarsh-Rose model is a neuron model. It is composed of 3 differential equations and can generate several firing patterns by tuning patterns.

# %%
import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt

# %%
bp.math.set_dt(dt=0.01)
bp.set_default_odeint('rk4')

# %%
types = ['quiescence', 'spiking', 'bursting', 'irregular_spiking', 'irregular_bursting']
bs = bp.math.array([1.0, 3.5, 2.5, 2.95, 2.8])
Is = bp.math.array([2.0, 5.0, 3.0, 3.3, 3.7])

# %%
# define neuron type
group = brainmodels.neurons.HindmarshRose(len(types), b=bs, monitors=['V'])
group = bp.math.jit(group)
group.run(1e3, inputs=['input', Is], report=0.1)

# %%
fig, gs = bp.visualize.get_figure(row_num=3, col_num=2, row_len=3, col_len=5)
for i, mode in enumerate(types):
    # plot
    fig.add_subplot(gs[i // 2, i % 2])
    plt.plot(group.mon.ts, group.mon.V[:, i])
    plt.title(mode)
    plt.xlabel('Time [ms]')
plt.show()
