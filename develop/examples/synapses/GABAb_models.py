# -*- coding: utf-8 -*-

import sys


import brainpy as bp
#bp.backend.set('numba')
import matplotlib.pyplot as plt
import brainmodels


class LIF2(bp.NeuGroup):


    target_backend = 'general'

    @staticmethod
    def derivative(V, t, Iext, V_rest, R, tau):
        return (-V + V_rest + R * Iext) / tau

    def __init__(self, size, t_refractory=1., V_rest=0.,
                 V_reset=-5., V_th=20., R=1., tau=10., **kwargs):
        # parameters
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.t_refractory = t_refractory

        # variables
        num = bp.size2len(size)
        self.t_last_spike = bp.backend.ones(num) * -1e7
        self.input = bp.backend.zeros(num)
        self.V = bp.backend.ones(num) * V_reset
        self.refractory = bp.backend.zeros(num, dtype=bool)
        self.spike = bp.backend.zeros(num, dtype=bool)

        self.int_V = bp.odeint(self.derivative)
        super(LIF2, self).__init__(size=size, **kwargs)

    def update(self, _t):
        refractory = (_t - self.t_last_spike) <= self.t_refractory
        V = self.int_V(self.V, _t, self.input, self.V_rest, self.R, self.tau)
        V = bp.backend.where(refractory, self.V, V)
        spike = V >= self.V_th
        self.t_last_spike = bp.backend.where(spike, _t, self.t_last_spike)
        self.V = bp.backend.where(spike, self.V_reset, V)
        self.refractory = refractory
        self.input[:] = 0.
        self.spike = spike



duration = 100.
dt = 0.02
print(bp.__version__)
print(brainmodels.__version__)
bp.backend.set('numpy', dt=dt)
size = 10
neu_pre = LIF2(size, monitors = ['V', 'input', 'spike'])
neu_pre.V_rest = -65.
neu_pre.V_reset = -70.
neu_pre.V_th = -50.
neu_pre.V = bp.backend.ones(size) * -65.
neu_post = LIF2(size, monitors = ['V', 'input', 'spike'])

syn_GABAb = brainmodels.tensor_backend.synapses.GABAb1(pre = neu_pre, post = neu_post, conn = bp.connect.One2One(), delay = 0., monitors = ['s'])

#current, dur = bp.inputs.constant_current([(21., 20.), (0., duration - 20.)])
net = bp.Network(neu_pre, syn_GABAb, neu_post)
net.run(200, inputs = [(neu_pre, 'input', 25)], report = True)



# paint gabaa
ts = net.ts
fig, gs = bp.visualize.get_figure(2, 1, 5, 6)

#print(gabaa.mon.s.shape)
fig.add_subplot(gs[0, 0])
plt.plot(ts, syn_GABAb.mon.s[:, 0, 0], label='s')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(ts, neu_post.mon.V[:, 0], label='post.V')
plt.legend()

plt.show()
