# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
from numba import prange

bp.integrators.set_default_odeint('rk4')
bp.backend.set(backend='numba', dt=0.01)

class Alpha_vec(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    def __init__(self, pre, post, conn, delay=0., tau=2.0, **kwargs):
        # parameters
        self.tau = tau
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

        super(Alpha_vec, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def integral(s, x, t, tau):
        dxdt = (-2 * tau * x - s) / (tau ** 2)
        dsdt = x
        return dsdt, dxdt

    
    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]

            self.s[i], self.x[i] = self.integral(self.s[i], self.x[i], _t, self.tau)
            self.x[i] += self.pre.spike[pre_id]

            self.out.push(i, self.w[i] * self.s[i])
            
            # output
            post_id = self.post_ids[i]
            self.post.input[post_id] += self.out.pull(i) 