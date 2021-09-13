# -*- coding: utf-8 -*-
# %% [markdown]
# # Example of Adaptive Quadratic Integrate-and-Fire model

# %%
import brainpy as bp
import brainmodels


# %%
group = brainmodels.neurons.AdQuaIF(1, monitors=['V', 'w'])
group.run(300, inputs=('input', 30.))

# %%
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(group.mon.ts, group.mon.V, ylabel='V')
fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(group.mon.ts, group.mon.w, ylabel='w', show=True)
