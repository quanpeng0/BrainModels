# -*- coding: utf-8 -*-
# %% [markdown]
# # Simple Example of Current-based Exponential Synapse
# %%
import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt

# %%
neu1 = brainmodels.neurons.LIF(1)
neu2 = brainmodels.neurons.LIF(1)
syn1 = brainmodels.synapses.ExpCUBA(neu1, neu2, bp.connect.All2All(), g_max=5.)
net = bp.Network(pre=neu1, syn=syn1, post=neu2)

runner = bp.StructRunner(net, inputs=[('pre.input', 25.)], monitors=['pre.V', 'post.V', 'syn.g'])
runner.run(150.)

# %%
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
plt.legend()
plt.show()
