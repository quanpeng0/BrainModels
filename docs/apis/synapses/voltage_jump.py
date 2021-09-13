# -*- coding: utf-8 -*-
# %% [markdown]
# # Simple Example for Voltage-jump Synapse

# %%
import brainpy as bp
import brainmodels

# %%
import matplotlib.pyplot as plt

# %%
neu1 = brainmodels.neurons.LIF(1, monitors=['V', 'spike'])
neu2 = brainmodels.neurons.LIF(1, monitors=['V'])
syn1 = brainmodels.synapses.VoltageJump(pre=neu1, post=neu2, conn=bp.connect.All2All(), delay=2.0)
net = bp.Network(neu1=neu1, syn1=syn1, neu2=neu2)
net.run(150., inputs=[('neu1.input', 25.), ('neu2.input', 10.)])

# %%
fig, gs = bp.visualize.get_figure(1, 1, 3, 8)
plt.plot(neu1.mon.ts, neu1.mon.V, label='pre-V')
plt.plot(neu1.mon.ts, neu2.mon.V, label='post-V')
plt.xlim(40, 150)
plt.legend()
plt.show()
