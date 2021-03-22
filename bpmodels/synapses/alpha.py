# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
from numba import prange

bp.integrators.set_default_odeint('rk4')
bp.backend.set(backend='numba', dt=0.01)

class Alpha_vec(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    def __init__(self, pre, post, conn, delay=0., g_max=0.20, E=0., tau=2.0, **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.tau = tau
        self.delay = delay

        # connections (requires)
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # data （ST)
        self.s = bp.backend.zeros(self.size)
        self.x = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)


        super(Alpha_vec, self).__init__(steps=[self.update, ],
                                        pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def integral(s, x, t, tau):
        dxdt = (-2 * tau * x - s) / (tau ** 2)
        dsdt = x
        return dsdt, dxdt

    # update and output
    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]

            self.s[i], self.x[i] = self.integral(self.s[i], self.x[i], _t, self.tau)
            self.x[i] += self.pre.spike[pre_id]

            # output
            self.g.push(i, self.g_max * self.s[i])

            post_id = self.post_ids[i]

            # COBA: post['input'] -= g * (post['V'] - E)
            # self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)

            # CUBA: post['input'] += ST['g']
            self.post.input[post_id] += self.g.pull(i) 