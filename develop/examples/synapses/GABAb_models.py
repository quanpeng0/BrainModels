# -*- coding: utf-8 -*-

import brainpy as bp
import matplotlib.pyplot as plt
import brainmodels

bp.backend.set('numba')
duration = 100.
dt = 0.02

size = 10

neu_pre = brainmodels.tensor_backend.neurons.LIF(size, monitors=['V', 'input', 'spike'], )
neu_pre.V_rest = -65.
neu_pre.V_reset = -70.
neu_pre.V_th = -50.
neu_pre.V = bp.backend.ones(size) * -65.

neu_post = brainmodels.tensor_backend.neurons.LIF(size, monitors=['V', 'input', 'spike'], )

syn_GABAb = brainmodels.tensor_backend.synapses.GABAb1(pre=neu_pre, post=neu_post, conn=bp.connect.One2One(),
                                                       delay=10., monitors=['s'], )

I, dur = bp.inputs.constant_current([(25, 20), (0, 1000)])
net = bp.Network(neu_pre, syn_GABAb, neu_post)
net.run(dur, inputs=[(neu_pre, 'input', I)], report=True)

# paint gabaa
ts = net.ts
plt.plot(ts, syn_GABAb.mon.s[:, 0, 0], label='s')
plt.legend()
plt.show()
