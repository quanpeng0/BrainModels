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
        self.pre_ids, self.post_ids = self.conn.requires(
            'pre_ids', 'post_ids'
        )
        self.size = len(self.pre_ids)

        # data
        self.w = bp.ops.ones(self.size) * 0.05

        self.integral = bp.odeint(f=self.derivative)
        super(Oja, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        post_r = bp.ops.zeros(self.post.size[0])
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            post_id = self.post_ids[i]
            add = self.w[i] * self.pre.r[pre_id]
            post_r[post_id] += add
            self.w[i] = self.integral(
                self.w[i], _t, self.gamma,
                self.pre.r[pre_id], self.post.r[post_id])
        self.post.r = post_r
