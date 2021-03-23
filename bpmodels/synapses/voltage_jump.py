# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
from numba import prange

bp.integrators.set_default_odeint('rk4')
bp.backend.set(backend='numba', dt=0.01)

class Voltage_jump(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    def __init__(self, pre, post, conn, delay=0., post_refractory=False,  **kwargs):
        # parameters
        self.delay = delay
        self.post_refractory = post_refractory

        # connections (requires)
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # data （ST)
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(Voltage_jump, self).__init__(
                                        pre=pre, post=post, **kwargs)

    # @staticmethod

    # update and output
    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            self.s[i] = self.pre.spike[pre_id]

            # output
            post_id = self.post_ids[i]
            if self.post_refractory:
                self.g.push(i, self.s[i] * (1. - self.post.refractory[post_id]))
            else:
                self.g.push(i, self.s[i])
            
            self.post.V[post_id] += self.g.pull(i)