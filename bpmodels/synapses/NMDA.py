# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
from numba import prange

bp.integrators.set_default_odeint('rk4')
bp.backend.set(backend='numba', dt=0.01)

class NMDA_vec(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    def __init__(self, pre, post, conn, delay=0., g_max=0.15, E=0., cc_Mg=1.2,
                    alpha=0.062, beta=3.57, tau=100, a=0.5, tau_rise = 2., **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.cc_Mg = cc_Mg
        self.tau = tau
        self.tau_rise = tau_rise
        self.a = a
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # variables
        self.s = bp.backend.zeros(self.size)
        self.x = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)


        super(NMDA_vec, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def integral(s, x, t, tau_rise, tau_decay, a):
        dxdt = -x / tau_rise
        dsdt = -s / tau_decay + a * x * (1 - s)
        return dsdt, dxdt

    
    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]

            self.x[i] += self.pre.spike[pre_id]
            self.s[i], self.x[i] = self.integral(self.s[i], self.x[i], _t, self.tau_rise, self.tau, self.a)

            # output
            self.g.push(i, self.g_max * self.s[i])

            post_id = self.post_ids[i]

            g_inf = 1 + self.cc_Mg / self.beta * bp.backend.exp(-self.alpha * self.post.V[post_id])

            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E) / g_inf
