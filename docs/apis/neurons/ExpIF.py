# -*- coding: utf-8 -*-
# %% [markdown]
# # Example of Exponential Integrate-and-Fire model

# %%
import brainpy as bp
import brainmodels

# %%
group = brainmodels.neurons.ExpIF(1, monitors=['V'])
group.run(300., inputs=('input', 10.))

# %%
bp.visualize.line_plot(group.mon.ts, group.mon.V, ylabel='V', show=True)
