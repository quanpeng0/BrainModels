# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
    'GABAb1',
    'GABAb2',
]


class GABAb1(bp.TwoEndConn):
    target_backend = 'general'

    @staticmethod
    def derivative(G, R, t, k1, k2, k3, k4, TT):
        dGdt = k1 * R - k2 * G
        dRdt = k3 * TT * (1 - R) - k4 * R
        return dGdt, dRdt

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
        self.g = self.register_constant_delay('g', size=self.size,
                                              delay_time=delay)
        self.t_last_pre_spike = bp.backend.ones(self.size) * -1e7

        self.integral = bp.odeint(f=self.derivative, method='euler')
        super(GABAb1, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        spike = bp.backend.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.t_last_pre_spike = bp.backend.where(spike, _t,
                                                 self.t_last_pre_spike)
        TT = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
        self.G, self.R = self.integral(self.G, self.R, _t,
                                       self.k1, self.k2, self.k3, self.k4, TT)
        self.s = self.G ** 4 / (self.G ** 4 + self.kd)
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.backend.sum(self.g.pull(), 0) \
                           * (self.post.V - self.E)


class GABAb2(bp.TwoEndConn):
    target_backend = 'general'

    @staticmethod
    def derivative(R, D, G, t, k1, k2, k3, TT, k4, k5, k6):
        dRdt = k1 * TT * (1 - R - D) - k2 * R + k3 * D
        dDdt = k4 * R - k3 * D
        dGdt = k5 * R - k6 * G
        return dRdt, dDdt, dGdt

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
        self.g = self.register_constant_delay('g', size=self.size,
                                              delay_time=delay)
        self.t_last_pre_spike = bp.backend.ones(self.size) * -1e7

        self.integral = bp.odeint(f=self.derivative, method='euler')
        super(GABAb2, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        spike = bp.backend.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.t_last_pre_spike = bp.backend.where(spike, _t,
                                                 self.t_last_pre_spike)
        T = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
        self.R, self.D, self.G = self.integral(
            self.R, self.D, self.G, _t,
            self.k1, self.k2, self.k3, T,
            self.k4, self.k5, self.k6)
        self.s = (self.G ** 4 / (self.G ** 4 + self.kd))
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.backend.sum(self.g.pull(), axis=0) \
                           * (self.post.V - self.E)
