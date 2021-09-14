# -*- coding: utf-8 -*-
# %% [markdown]
# # Simple Example of Current-based Exponential Synapse
# %%
import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt

# %%
neu1 = brainmodels.neurons.LIF(1, monitors=['V'], name='X')
neu2 = brainmodels.neurons.LIF(1, monitors=['V'])
syn1 = brainmodels.synapses.ExpCUBA(neu1, neu2, bp.connect.All2All(), g_max=5, monitors=['g'])
net = bp.Network(neu1, syn1, neu2)
net.run(150., inputs=[('X.input', 25.)])

# %%
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
plt.plot(neu1.mon.ts, neu1.mon.V, label='pre-V')
plt.plot(neu2.mon.ts, neu2.mon.V, label='post-V')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(neu1.mon.ts, syn1.mon.g, label='g')
plt.legend()
plt.show()
