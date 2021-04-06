# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import brainpy as bp
import numpy as np
import brainmodels

dt = 0.02
bp.backend.set(backend='numpy', dt=dt)

# Set pre & post NeuGroup
pre = brainmodels.neurons.LIF(10, monitors=['V', 'input', 'spike'])
pre.V = -65. * np.ones(pre.V.shape)
post = brainmodels.neurons.LIF(10, monitors=['V', 'input', 'spike'])
post.V = -65. * np.ones(pre.V.shape)

# Set synapse connection & network
syn = brainmodels.synapses.Exponential(pre=pre, post=post,
                                       conn=bp.connect.All2All(), monitors=['s'], delay=10.)
net = bp.Network(pre, syn, post)

(current, duration) = bp.inputs.constant_current([(0, 15), (30, 15), (0, 70)])
net.run(duration=duration, inputs=(pre, 'input', current), report=True)

# Figure
ts = net.ts
plt.plot(ts, syn.mon.s[:, 0, 0], label='s')
plt.ylabel('Conductance (µmhos)')
plt.xlabel('Time (ms)')
plt.legend()
plt.show()
