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
two_exponentials = brainmodels.synapses.Two_exponentials(pre=pre, post=post,
                                    conn=bp.connect.All2All(), monitors=['s'], delay=10.)
net = bp.Network(pre, two_exponentials, post)

(current, duration) = bp.inputs.constant_current([(0, 25), (30, 5), (0, 170)])
net.run(duration=duration, inputs=(pre, 'input', current), report=True)

# Figure
ts = net.ts
plt.plot(ts, two_exponentials.mon.s[:, 0, 0], label='s')
plt.ylabel('Conductance (µmhos)')
plt.xlabel('Time (ms)')
plt.legend()
plt.show()
