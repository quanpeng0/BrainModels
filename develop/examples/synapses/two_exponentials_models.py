# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import brainpy as bp
import brainmodels

backend = 'numpy'
bp.backend.set(backend=backend, dt=.02)
brainmodels.set_backend(backend=backend)

# Set pre & post NeuGroup
pre = brainmodels.neurons.LIF(1, monitors=['V', 'input', 'spike'])
pre.V = -65. * bp.ops.ones(pre.V.shape)
post = brainmodels.neurons.LIF(1, monitors=['V', 'input', 'spike'])
post.V = -65. * bp.ops.ones(pre.V.shape)

# Set synapse connection & network
syn = brainmodels.synapses.Two_exponentials(pre=pre, post=post,
                                            conn=bp.connect.All2All(), monitors=['s'], delay=10.)
net = bp.Network(pre, syn, post)

(current, duration) = bp.inputs.constant_current([(0, 15), (30, 15), (0, 70)])
net.run(duration=duration, inputs=(pre, 'input', current), report=True)

# Figure
ts = net.ts
plt.plot(ts, syn.mon.s[:, 0], label='s')
plt.ylabel('s')
plt.xlabel('Time (ms)')
plt.legend()
plt.show()
