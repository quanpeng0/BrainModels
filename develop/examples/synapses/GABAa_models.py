# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import brainpy as bp
import numpy as np
import bpmodels
from bpmodels.neurons import get_LIF

duration = 500.
dt = 0.02
bp.profile.set(jit=True, dt=dt, merge_steps=True)
LIF_neuron = get_LIF()
GABAa_syn = bpmodels.synapses.get_GABAa1(mode='matrix')

# build and simulate gabaa net
pre = bp.NeuGroup(LIF_neuron, geometry=(10,), monitors=['V', 'input', 'spike'])
pre.pars['V_rest'] = -65.
pre.ST['V'] = -65.
post = bp.NeuGroup(LIF_neuron, geometry=(10,), monitors=['V', 'input', 'spike'])
post.pars['V_rest'] = -65.
post.ST['V'] = -65.

gabaa = bp.SynConn(model=GABAa_syn, pre_group=pre, post_group=post,
                   conn=bp.connect.One2One(), monitors=['s'], delay=10.)
gabaa.set_schedule(['input', 'update', 'output', 'monitor'])

net = bp.Network(pre, gabaa, post)

current = bp.inputs.spike_current([10, 110, 210, 300, 305, 310, 315, 320],
                                  bp.profile._dt, 1., duration=duration)
net.run(duration=duration, inputs=[gabaa, 'pre.spike', current, "="], report=True)

# paint gabaa
ts = net.ts
fig, gs = bp.visualize.get_figure(2, 2, 5, 6)

print(gabaa.mon.s.shape)
fig.add_subplot(gs[0, 0])
plt.plot(ts, gabaa.mon.s[:, 0], label='s')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(ts, post.mon.V[:, 0], label='post.V')
plt.legend()

fig.add_subplot(gs[0, 1])
plt.plot(ts, post.mon.input[:, 0], label='post.input')
plt.legend()

plt.show()
