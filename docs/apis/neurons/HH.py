# -*- coding: utf-8 -*-
# %% [markdown]
# # Example of Hodgkin–Huxley model
# %%
import brainpy as bp
import brainmodels


# %%
def run_hh1():
  group = bp.math.jit(brainmodels.neurons.HH(2, monitors=['V']))

  group.run(200., inputs=('input', 10.), report=0.1)
  bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

  group.run(200., report=0.1)
  bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)


# %%
run_hh1()


# %%
def run_hh2():
  group = bp.math.jit(brainmodels.neurons.HH(2, monitors=bp.Monitor(variables=['V'], intervals=[1.])))

  group.run(200., inputs=('input', 10.), report=0.1)
  bp.visualize.line_plot(group.mon['V.t'], group.mon.V, show=True)


# %%
run_hh2()
