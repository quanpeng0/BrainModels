# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
from numba import prange

bp.integrators.set_default_odeint('rk4')
bp.backend.set(backend='numba', dt=0.01)

class Two_exponentials_vec(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    def __init__(self, pre, post, conn, delay=0., tau1=1.0, tau2=3.0, **kwargs):
        # parameters
        self.tau1 = tau1
        self.tau2 = tau2
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # variables
        self.s = bp.backend.zeros(self.size)
        self.x = bp.backend.zeros(self.size)
        self.w = bp.backend.ones(self.size) * .2
        self.out = self.register_constant_delay('out', size=self.size, delay_time=delay)

        super(Two_exponentials_vec, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def integral(s, x, t, tau1, tau2):
        dxdt = (-(tau1 + tau2) * x - s) / (tau1 * tau2)
        dsdt = x
        return dsdt, dxdt
    
    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]

            self.s[i], self.x[i] = self.integral(self.s[i], self.x[i], _t, self.tau1, self.tau2)
            self.x[i] += self.pre.spike[pre_id]

            self.out.push(i, self.w[i] * self.s[i])
            
            # output
            post_id = self.post_ids[i]
            self.post.input[post_id] += self.out.pull(i)            self.post.input[post_id] += self.g.pull(i) 