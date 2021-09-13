# -*- coding: utf-8 -*-
# %% [markdown]
# # Example of Morris-Lecar model
# %%
import brainpy as bp
import brainmodels

# %%
bp.math.set_dt(0.05)
bp.integrators.set_default_odeint('rk4')

# %%
group = brainmodels.neurons.MorrisLecar(1, monitors=['V', 'W'])
group.run(1000, inputs=('input', 100.))

# %%
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(group.mon.ts, group.mon.W, ylabel='W')
fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(group.mon.ts, group.mon.V, ylabel='V', show=True)

