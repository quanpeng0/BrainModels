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


# %%
import matplotlib.pyplot as plt

def run_hh3():
  group = bp.math.jit(brainmodels.neurons.HH(2, monitors=['V']))

  I1 = bp.inputs.spike_input(sp_times=[500., 550., 1000, 1030, 1060, 1100, 1200], sp_lens=5, sp_sizes=5., duration=2000, )
  I2 = bp.inputs.spike_input(sp_times=[600.,       900, 950, 1500], sp_lens=5, sp_sizes=5., duration=2000, )
  I1 += bp.math.random.normal(0, 3, size=I1.shape)
  I2 += bp.math.random.normal(0, 3, size=I2.shape)
  I = bp.math.stack((I1, I2), axis=-1)
  group.run(2000., inputs=('input', I, 'iter'), report=0.1)

  fig, gs = bp.visualize.get_figure(1, 1, 2, 10)
  fig.add_subplot(gs[0, 0])
  plt.plot(group.mon.ts, group.mon.V[:, 0])
  plt.plot(group.mon.ts, group.mon.V[:, 1] + 130)
  plt.xlim(10, 2000)
  plt.xticks([])
  plt.yticks([])
  plt.show()
