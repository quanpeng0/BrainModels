# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
from numba import prange

bp.integrators.set_default_odeint('rk4')
bp.backend.set(backend='numba', dt=0.01)

class Two_exponentials_vec(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    def __init__(self, pre, post, conn, delay=0., g_max=0.20, E=0., tau1=1.0, tau2=3.0, **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.tau1 = tau1
        self.tau2 = tau2
        self.delay = delay

        # connections (requires)
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # data （ST)
        self.s = bp.backend.zeros(self.size)
        self.x = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)


        super(Two_exponentials_vec, self).__init__(
                                        pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def integral(s, x, t, tau1, tau2):
        dxdt = (-(tau1 + tau2) * x - s) / (tau1 * tau2)
        dsdt = x
        return dsdt, dxdt

    # update and output
    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]

            self.s[i], self.x[i] = self.integral(self.s[i], self.x[i], _t, self.tau1, self.tau2)
            self.x[i] += self.pre.spike[pre_id]

            # output
            self.g.push(i, self.g_max * self.s[i])

            post_id = self.post_ids[i]

            # COBA: post['input'] -= g * (post['V'] - E)
            # self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)

            # CUBA: post['input'] += ST['g']
            self.post.input[post_id] += self.g.pull(i) 