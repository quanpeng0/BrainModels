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

    def __init__(self, size, V_rest = V_rest, V_reset = V_reset,
                 V_th = V_th, tau = tau, **kwargs):
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.tau = tau

        self.V = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)
        self.spike = bp.backend.zeros(size, dtype = bool)

        super(LIF, self).__init__(size = size, **kwargs)

    @staticmethod
    @bp.odeint(method = 'rk4')
    def integral(V, t, I_ext, V_rest, tau):
        return (-V + V_rest + I_ext) / tau

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

    def __init__(self, pre, post, conn,
                 tau_decay = tau_decay, w = 0., 
                 **kwargs):
        self.tau_decay = tau_decay
        self.w = w

        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        self.s = bp.backend.zeros(self.size)
        self.g = bp.backend.zeros(self.size)

        super(Syn, self).__init__(pre = pre, post = post, **kwargs)
    
    @staticmethod
    @bp.odeint(method = 'rk4')
    def integral(s, t, tau):
        return -s / tau

    def update(self, _t):
        for i in prange(self.size):
            self.s[i] = self.integral(self.s[i], _t, self.tau_decay)
            pre_id = self.pre_ids[i]
            self.s[i] += self.pre.spike[pre_id]
            g = self.w * self.s[i]
            post_id = self.post_ids[i]
            self.post.input[post_id] += g

neu_E = LIF(N_E, monitors = ['spike'])
neu_I = LIF(N_I, monitors = ['spike'])
neu_E.V = V_rest + np.random.random(N_E) * (V_th - V_rest)
neu_I.V = V_rest + np.random.random(N_I) * (V_th - V_rest)

syn_E2E = Syn(pre = neu_E, post = neu_E,
              conn = bp.connect.FixedProb(prob = prob))
syn_E2I = Syn(pre = neu_E, post = neu_I,
              conn = bp.connect.FixedProb(prob = prob))
syn_I2E = Syn(pre = neu_I, post = neu_E,
              conn = bp.connect.FixedProb(prob = prob))
syn_I2I = Syn(pre = neu_I, post = neu_I,
              conn = bp.connect.FixedProb(prob = prob))
syn_E2E.w = JE
syn_E2I.w = JE
syn_I2E.w = -JI
syn_I2I.w = -JI

net = bp.Network(neu_E, neu_I, 
                 syn_E2E, syn_E2I, 
                 syn_I2E, syn_I2I)
net.run(500., inputs = [(neu_E, 'input', 3.), (neu_I, 'input', 3.)], report = True)


fig, gs = bp.visualize.get_figure(4, 1, 2, 10)
fig.add_subplot(gs[:3, 0])
bp.visualization.raster_plot(net.ts, neu_E.mon.spike)

fig.add_subplot(gs[3, 0])
rate = bp.measure.firing_rate(neu_E.mon.spike, 5.)
plt.plot(net.ts, rate)
plt.show()


"""
Implementation of E/I balance network.
"""
'''
import brainpy as bp
import numpy as np
from numba import prange
import matplotlib.pyplot as plt
import pdb

num_exc = 500
num_inh = 500
prob = 0.1

bp.backend.set('numpy')

# -------
# neuron
# -------
V_rest = -52.
V_th = -50.
V_reset = -60.
tau = 10.

class LIF(bp.NeuGroup):
    target_backend = 'general'

    def __init__(self, size, V_rest = -52., V_th = -50., 
                 V_reset = -60., tau = 10., **kwargs):
        #params
        self.V_rest = V_rest
        self.V_th = V_th
        self.V_reset = V_reset
        self.tau = tau

        #vars
        self.V = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)
        self.spike = bp.backend.zeros(size, dtype = bool)

        super(LIF, self).__init__(size = size, **kwargs)

    @staticmethod
    @bp.odeint(method = 'exponential_euler')
    def integral(V, t, I_ext, V_rest, tau):
        return (-V + V_rest + I_ext) / tau

    def update(self, _t):
        V = self.integral(
            self.V, _t, self.input, 
            self.V_rest, self.tau)
        sp = (V > self.V_th)
        V[sp] = self.V_reset
        self.spike = sp
        self.V = V
        self.input[:] = 0.

neu_E = LIF(num_exc, monitors=['spike'], show_code=True)
neu_I = LIF(num_inh, monitors=['spike'], show_code=True)
neu_E.V = np.random.random(num_exc) * (V_th - V_rest) + V_rest
neu_I.V = np.random.random(num_inh) * (V_th - V_rest) + V_rest

# -------
# synapse
# -------

class Exp(bp.TwoEndConn):
    target_backend = 'general'

    def __init__(self, pre, post, conn, 
                 tau = 2., w = 0., **kwargs):
        self.tau = tau
        self.w = w

        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        self.s = bp.backend.zeros(self.size)
        
        super(Exp, self).__init__(pre = pre, post = post, **kwargs)

    @staticmethod
    @bp.odeint()
    def integral(s, t, tau):
        return - s / tau

    def update(self, _t):
        self.s = self.integral(self.s, _t, self.tau)
        self.s += bp.backend.reshape(
            self.pre.spike, (-1, 1)) * self.conn_mat
        self.post.input += bp.backend.sum(self.w * self.s, axis = 0)

JE = 1 / np.sqrt(prob * num_exc)
JI = 1 / np.sqrt(prob * num_inh)

syn_E2E = Exp(pre=neu_E, post=neu_E,
              conn=bp.connect.FixedProb(prob=prob),
              w = JE)
syn_E2I = Exp(pre=neu_E, post=neu_I,
              conn=bp.connect.FixedProb(prob=prob),
              w = JE)

syn_I2E = Exp(pre=neu_I, post=neu_E,
              conn=bp.connect.FixedProb(prob=prob),
              w = -JI)
syn_I2I = Exp(pre=neu_I, post=neu_I,
              conn=bp.connect.FixedProb(prob=prob),
              w = -JI)

net = bp.Network(neu_E, neu_I, syn_E2E, syn_E2I, 
                 syn_I2E, syn_I2I)
net.run(duration=1000., 
        inputs=[(neu_E, 'input', 3.), (neu_I, 'input', 3.)], 
        report=True,
        report_percent = 0.01)

# --------------
# visualization
# --------------

fig, gs = bp.visualize.get_figure(4, 1, 2, 10)

fig.add_subplot(gs[:3, 0])
bp.visualize.raster_plot(net.ts, neu_E.mon.spike)

fig.add_subplot(gs[3, 0])
rates = bp.measure.firing_rate(neu_E.mon.spike, 5.)
plt.plot(net.ts, rates)
plt.show()
'''
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