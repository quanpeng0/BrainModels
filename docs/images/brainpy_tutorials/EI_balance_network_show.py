# -*- coding: utf-8 -*-
"""
Implementation of E/I balance network.
"""

import brainpy as bp
import numpy as np
from numba import prange
import matplotlib.pyplot as plt

num_exc = 500
num_inh = 500
prob = 0.1

# -------
# neuron
# -------


tau = 10.
V_rest = -52.
V_reset = -60.
V_threshld = -50.

class LIF(bp.NeuGroup):
    target_backend = ['numpy', 'numba']

    def __init__(self, size, tau = 10., V_rest = -52., 
                 V_reset = -60., V_th = -50., 
                 **kwargs):
        #params
        self.tau = tau
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th

        #vars
        self.V = bp.backend.zeros(size)
        self.spike = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)

        super(LIF, self).__init__(size = size, **kwargs)
    
    @staticmethod
    @bp.odeint()
    def integral(V, t, I_ext, V_rest, tau):
        return (- V + V_rest + I_ext) / tau

    def update(self, _t):
        V = self.integral(self.V, _t, self.input, self.V_rest, self.tau)    
        for i in prange(self.size[0]):
            if V[i] > self.V_th:
                V[i] = self.V_reset
                self.spike[i] = 1.
            else:
                self.spike[i] = 0.
        self.V = V
        self.input[:] = 0.


# -------
# synapse
# -------

class Exp(bp.TwoEndConn):
    target_backend = ['numpy', 'numba']

    def __init__(self, pre, post, conn, delay = 0., 
                 tau = 2., **kwargs):
        #params
        self.tau = tau
        self.delay = delay

        #conns
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        #data
        self.s = bp.backend.zeros(self.size)
        self.w = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(Exp, self).__init__(pre = pre, post = post, **kwargs)

    @staticmethod
    @bp.odeint()
    def integral(s, t, tau):
        return -s / tau
    
    def update(self, _t):
        for i in prange(self.size):
            self.s[i] = self.integral(self.s[i], _t, self.tau)
            pre_id = self.pre_ids[i]
            post_id = self.post_ids[i]
            self.s[i] += self.pre.spike[pre_id]
            self.g.push(i, self.w * self.s[i])
            self.post.input[post_id] += self.g.pull(i)


bp.backend.set('numpy')

neu_E = LIF(num_exc, monitors=['spike'])
neu_I = LIF(num_inh, monitors=['spike'])
neu_E.V = np.random.random(num_exc) * (V_threshld - V_rest) + V_rest
neu_I.V = np.random.random(num_inh) * (V_threshld - V_rest) + V_rest

JE = 1 / np.sqrt(prob * num_exc)
JI = 1 / np.sqrt(prob * num_inh)

syn_E2E = Exp(pre=neu_E, post=neu_E,
              conn=bp.connect.FixedProb(prob=prob))
syn_E2E.w = JE
syn_E2I = Exp(pre=neu_E, post=neu_I,
              conn=bp.connect.FixedProb(prob=prob))
syn_E2I.w = JE

syn_I2E = Exp(pre=neu_I, post=neu_E,
              conn=bp.connect.FixedProb(prob=prob))
syn_I2E.w = -JI
syn_I2I = Exp(pre=neu_I, post=neu_E,
              conn=bp.connect.FixedProb(prob=prob))
syn_I2I.w = -JI

net = bp.Network(neu_E, neu_I, 
                 syn_E2E, syn_E2I, 
                 syn_I2E, syn_I2I)
net.run(duration=500., 
        inputs=[(neu_E, 'input', 3.), (neu_I, 'input', 3.)], 
        report=True)

# --------------
# visualization
# --------------

fig, gs = bp.visualize.get_figure(4, 1, 2, 10)

fig.add_subplot(gs[:3, 0])
bp.visualize.raster_plot(net.ts, neu_E.mon.spike, xlim=(50, 450))

fig.add_subplot(gs[3, 0])
rates = bp.measure.firing_rate(neu_E.mon.spike, 5.)
plt.plot(net.ts, rates)
plt.xlim(50, 450)
plt.show()
'''
"""
Implementation of E/I balance network.
"""


import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt


bp.profile.set(jit=True, device='cpu',
               numerical_method='exponential')

num_exc = 500
num_inh = 500
prob = .1

# -------
# neuron
# -------


tau = 10.
V_rest = -52.
V_reset = -60.
V_threshld = -50.


@bp.integrate
def int_f(V, t, Isyn):
    return (-V + V_rest + Isyn) / tau


def update(ST, _t):
    V = int_f(ST['V'], _t, ST['input'])
    if V >= V_threshld:
        ST['spike'] = 1.
        V = V_reset
    else:
        ST['spike'] = 0.
    ST['V'] = V
    ST['input'] = 0.


neu = bp.NeuType(name='LIF',
                 ST=bp.types.NeuState({'V': 0, 'spike': 0., 'input': 0.}),
                 steps=update,
                 mode='scalar')

# -------
# synapse
# -------


tau_decay = 2.
JE = 1 / np.sqrt(prob * num_exc)
JI = 1 / np.sqrt(prob * num_inh)

@bp.integrate
def ints(s, t):
    return - s / tau_decay


def update(ST, _t, pre):
    s = ints(ST['s'], _t)
    s += pre['spike']
    ST['s'] = s
    ST['g'] = ST['w'] * s


def output(ST, post):
    post['input'] += ST['g']


syn = bp.SynType(name='exponential_synapse',
                 ST=bp.types.SynState(['s', 'g', 'w']),
                 steps=(update, output),
                 mode='scalar')

# -------
# network
# -------

group = bp.NeuGroup(neu,
                    geometry=num_exc + num_inh,
                    monitors=['spike'])
group.ST['V'] = np.random.random(num_exc + num_inh) * (V_threshld - V_rest) + V_rest

exc_conn = bp.SynConn(syn,
                      pre_group=group[:num_exc],
                      post_group=group,
                      conn=bp.connect.FixedProb(prob=prob))
exc_conn.ST['w'] = JE

inh_conn = bp.SynConn(syn,
                      pre_group=group[num_exc:],
                      post_group=group,
                      conn=bp.connect.FixedProb(prob=prob))
inh_conn.ST['w'] = -JI

net = bp.Network(group, exc_conn, inh_conn)
net.run(duration=500., inputs=[(group, 'ST.input', 3.)], report=True)

# --------------
# visualization
# --------------

fig, gs = bp.visualize.get_figure(4, 1, 2, 10)

fig.add_subplot(gs[:3, 0])
bp.visualize.raster_plot(net.ts, group.mon.spike, xlim=(50, 450))

fig.add_subplot(gs[3, 0])
rates = bp.measure.firing_rate(group.mon.spike, 5.)
plt.plot(net.ts, rates)
plt.xlim(50, 450)
plt.show()

'''