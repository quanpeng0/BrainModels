# -*- coding: utf-8 -*-
# %% [markdown]
# # Example of LIF model
# %%
import brainmodels
import brainpy as bp

# %%
group = bp.math.jit(brainmodels.neurons.LIF(100, monitors=['V']))

# %%
group.run(duration=200., inputs=('input', 26.), report=0.1)
bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

# %%
group.run(duration=(200, 400.), report=0.1)
bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

