# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
    'GABAb1',
    'GABAb2',
]


class GABAb1(bp.TwoEndConn):
    target_backend = 'general'

    def __init__(self, pre, post, conn, delay=0.,
                 g_max=0.02, E=-95., k1=0.18, k2=0.034,
                 k3=0.09, k4=0.0012, kd=100.,
                 T=0.5, T_duration=0.3, **kwargs):
        self.g_max = g_max
        self.E = E
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.kd = kd
        self.T = T
        self.T_duration = T_duration
        self.delay = delay

        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        self.R = bp.backend.zeros(self.size)
        self.G = bp.backend.zeros(self.size)
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)
        self.t_last_pre_spike = bp.backend.ones(self.size) * -1e7

        super(GABAb1, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint
    def integral(G, R, t, k1, k2, k3, k4, TT):
        dGdt = k1 * R - k2 * G
        dRdt = k3 * TT * (1 - R) - k4 * R
        return dGdt, dRdt

    def update(self, _t):
        spike = bp.backend.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.t_last_pre_spike = bp.backend.where(spike, _t, self.t_last_pre_spike)
        TT = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
        self.G, self.R = self.integral(self.G, self.R, _t,
                                       self.k1, self.k2, self.k3, self.k4, TT)
        self.s = self.G ** 4 / (self.G ** 4 + self.kd)
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.backend.sum(self.g.pull(), 0) * (self.post.V - self.E)


class GABAb2(bp.TwoEndConn):
    target_backend = 'general'

    def __init__(self, pre, post, conn, delay=0.,
                 g_max=0.02, E=-95., k1=0.66, k2=0.02,
                 k3=0.0053, k4=0.017, k5=8.3e-5, k6=7.9e-3,
                 kd=100., T=0.5, T_duration=0.5,
                 **kwargs):
        # params
        self.g_max = g_max
        self.E = E
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.kd = kd
        self.T = T
        self.T_duration = T_duration

        # conns
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        # vars
        self.D = bp.backend.zeros(self.size)
        self.R = bp.backend.zeros(self.size)
        self.G = bp.backend.zeros(self.size)
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)
        self.t_last_pre_spike = bp.backend.ones(self.size) * -1e7

        super(GABAb2, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint
    def integral(R, D, G, t, k1, k2, k3, TT, k4, k5, k6):
        dRdt = k1 * TT * (1 - R - D) - k2 * R + k3 * D
        dDdt = k4 * R - k3 * D
        dGdt = k5 * R - k6 * G
        return dRdt, dDdt, dGdt

    def update(self, _t):
        spike = bp.backend.reshape(self.pre.spike, (-1, 1)) * self.conn_mat
        self.t_last_pre_spike = bp.backend.where(spike, _t, self.t_last_pre_spike)
        T = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
        self.R, self.D, self.G = self.integral(
            self.R, self.D, self.G, _t,
            self.k1, self.k2, self.k3, T,
            self.k4, self.k5, self.k6)
        self.s = (self.G ** 4 / (self.G ** 4 + self.kd))
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.backend.sum(self.g.pull(), axis=0) * (self.post.V - self.E)


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


import matplotlib.pyplot as plt

duration = 100.
dt = 0.02

size = 10
neu_pre = LIF2(size, monitors=['V', 'input', 'spike'], )
neu_pre.V_rest = -65.
neu_pre.V_reset = -70.
neu_pre.V_th = -50.
neu_pre.V = bp.backend.ones(size) * -65.
neu_post = LIF2(size, monitors=['V', 'input', 'spike'], )

syn_GABAb = GABAb1(pre=neu_pre, post=neu_post, conn=bp.connect.One2One(),
                   delay=10., monitors=['s'], )

I, dur = bp.inputs.constant_current([(25, 20), (0, 1000)])
net = bp.Network(neu_pre, syn_GABAb, neu_post)
net.run(dur, inputs=[(neu_pre, 'input', I)], report=True)

print(neu_pre.mon.spike.max())
print(syn_GABAb.mon.s.max())

bp.visualize.line_plot(net.ts, neu_pre.mon.V, show=True)

# paint gabaa
ts = net.ts
# print(gabaa.mon.s.shape)
plt.plot(ts, syn_GABAb.mon.s[:, 0, 0], label='s')
plt.legend()
plt.show()
