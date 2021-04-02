# -*- coding: utf-8 -*-
import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt
import numpy as np
from numba import prange

bp.backend.set('numba')

N_E = 500
N_I = 500
prob = 0.1

tau = 10.
V_rest = -52.
V_reset = -60.
V_th = -50.


class LIF(bp.NeuGroup):
    target_backend = 'general'

    @staticmethod
    def derivative(V, t, I_ext, V_rest, tau):
        dvdt = (-V + V_rest + I_ext) / tau
        return dvdt

    def __init__(self, size, V_rest=V_rest, V_reset=V_reset,
                 V_th=V_th, tau=tau, **kwargs):
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.tau = tau

        self.V = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)
        self.spike = bp.backend.zeros(size, dtype=bool)

        self.integral = bp.odeint(self.derivative, method='rk4')
        super(LIF, self).__init__(size=size, **kwargs)

    def update(self, _t):
        V = self.integral(self.V, _t, self.input, self.V_rest, self.tau)
        sp = V > self.V_th
        V[sp] = self.V_reset
        self.V = V
        self.spike = sp
        self.input[:] = 0.


tau_decay = 2.
JE = 1 / np.sqrt(prob * N_E)
JI = 1 / np.sqrt(prob * N_I)


class Syn(bp.TwoEndConn):
    target_backend = 'general'

    @staticmethod
    def derivative(s, t, tau_decay):
        dsdt = -s / tau_decay
        return dsdt

    def __init__(self, pre, post, conn,
                 tau_decay=tau_decay, w=0.,
                 **kwargs):
        self.tau_decay = tau_decay
        self.w = w

        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        self.s = bp.backend.zeros(self.size)
        self.g = bp.backend.zeros(self.size)

        self.integral = bp.odeint(self.derivative, method='exponential_euler')
        super(Syn, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        for i in prange(self.size):
            self.s[i] = self.integral(self.s[i], _t, self.tau_decay)
            pre_id = self.pre_ids[i]
            self.s[i] += self.pre.spike[pre_id]
            g = self.w * self.s[i]
            post_id = self.post_ids[i]
            self.post.input[post_id] += g


neu_E = LIF(N_E, monitors=['spike'])
neu_I = LIF(N_I, monitors=['spike'])
neu_E.V = V_rest + np.random.random(N_E) * (V_th - V_rest)
neu_I.V = V_rest + np.random.random(N_I) * (V_th - V_rest)

syn_E2E = Syn(pre=neu_E, post=neu_E,
              conn=bp.connect.FixedProb(prob=prob))
syn_E2I = Syn(pre=neu_E, post=neu_I,
              conn=bp.connect.FixedProb(prob=prob))
syn_I2E = Syn(pre=neu_I, post=neu_E,
              conn=bp.connect.FixedProb(prob=prob))
syn_I2I = Syn(pre=neu_I, post=neu_I,
              conn=bp.connect.FixedProb(prob=prob))
syn_E2E.w = JE
syn_E2I.w = JE
syn_I2E.w = -JI
syn_I2I.w = -JI

net = bp.Network(neu_E, neu_I,
                 syn_E2E, syn_E2I,
                 syn_I2E, syn_I2I)
net.run(500., inputs=[(neu_E, 'input', 3.), (neu_I, 'input', 3.)], report=True)

fig, gs = bp.visualize.get_figure(4, 1, 2, 10)
fig.add_subplot(gs[:3, 0])
bp.visualization.raster_plot(net.ts, neu_E.mon.spike)

fig.add_subplot(gs[3, 0])
rate = bp.measure.firing_rate(neu_E.mon.spike, 5.)
plt.plot(net.ts, rate)
plt.show()
