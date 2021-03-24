# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
from numba import prange

bp.integrators.set_default_odeint('rk4')
bp.backend.set(backend='numpy', dt=0.01)

class BCM(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    def __init__(self, pre, post, conn, lr=0.005, w_max=2., w_min=0., **kwargs):
        # parameters
        self.lr = lr
        self.w_max = w_max
        self.w_min = w_min
        self.dt = bp.backend._dt

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        # variables
        self.w = bp.backend.ones(self.size)
        self.sum_post_r = bp.backend.zeros(post.size[0])

        super(BCM, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint
    def int_w(w, t, lr, r_pre, r_post, r_th):
        dwdt = lr * r_post * (r_post - r_th) * r_pre
        return dwdt

    
    def update(self, _t):
        # update threshold
        self.sum_post_r += self.post.r
        r_th = self.sum_post_r / (_t / self.dt + 1) 

        # resize to matrix
        w = self.w * self.conn_mat
        dim = np.shape(w)
        r_th = np.vstack((r_th,)*dim[0])
        r_post = np.vstack((self.post.r,)*dim[0])
        r_pre = np.vstack((self.pre.r,)*dim[1]).T

        # update w
        w = self.int_w(w, _t, self.lr, r_pre, r_post, r_th)
        self.w = np.clip(w, self.w_min, self.w_max)

        # output
        self.post.r = np.dot(w.T, self.pre.r)
