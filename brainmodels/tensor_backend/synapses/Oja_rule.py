# -*- coding: utf-8 -*-
import brainpy as bp
import matplotlib.pyplot as plt
import numpy as np
from numba import prange


class Oja(bp.TwoEndConn):
    target_backend = 'general'

    @staticmethod
    def derivative(w, t, gamma, r_pre, r_post):
        dwdt = gamma * (r_post * r_pre - r_post * r_post * w)
        return dwdt

    def __init__(self, pre, post, conn, delay=0.,
                 gamma=0.005, w_max=1., w_min=0.,
                 **kwargs):
        # params
        self.gamma = gamma
        self.w_max = w_max
        self.w_min = w_min
        # no delay in firing rate models

        # conns
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        # data
        self.w = bp.ops.ones(self.size) * 0.05

        self.integral = bp.odeint(f=self.derivative)
        super(Oja, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        self.post.r = np.dot(self.pre.r, self.conn_mat * self.w)
        pre_mat_expand = np.expand_dims(self.pre.r, axis=1) \
            .repeat(self.post.r.shape[0], axis=1)
        post_mat_expand = np.expand_dims(self.post.r, axis=1) \
            .reshape(1, -1).repeat(self.pre.r.shape[0], axis=0)
        self.w = self.integral(self.w, _t, self.gamma, pre_mat_expand, post_mat_expand)
