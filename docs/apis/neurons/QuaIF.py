# -*- coding: utf-8 -*-
# %% [markdown]
# # Example of QuaIF model
# %%
import brainmodels
import brainpy as bp

# %%
group = brainmodels.neurons.QuaIF(1, monitors=['V'])

# %%
group.run(duration=200., inputs=('input', 20.), report=0.1)
bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

# %%
group.run(duration=(200, 400.), report=0.1)
bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

