# -*- coding: utf-8 -*-
# %% [markdown]
# # Simple Example of Conductance-based Dual Exponential Synapse
# %%

import sys
sys.path.append('/mnt/d/codes/Projects/BrainPy')
sys.path.append('/mnt/d/codes/Projects/BrainModels')

import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt

bp.math.use_backend('jax')


# %%
neu1 = brainmodels.neurons.HH(10, monitors=['V'], name='X')
neu2 = brainmodels.neurons.HH(10, monitors=['V'])
syn1 = brainmodels.synapses.DualExpCOBA(neu1, neu2, bp.connect.All2All(), E=0.,
                                        monitors=['g', 'h'])
net = bp.Network(neu1, syn1, neu2)
net.run(150., inputs=[('X.input', 5.)], report=0.1)

# %%
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(neu1.mon.ts, neu1.mon.V, legend='pre-V')
bp.visualize.line_plot(neu2.mon.ts, neu2.mon.V, legend='post-V')

fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(neu1.mon.ts, syn1.mon.g, legend='g')
bp.visualize.line_plot(neu1.mon.ts, syn1.mon.h, legend='h')
plt.legend()
plt.show()
