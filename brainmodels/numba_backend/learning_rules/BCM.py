# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np

bp.integrators.set_default_odeint('rk4')


class BCM(bp.TwoEndConn):
    """
    Bienenstock-Cooper-Munro (BCM) rule.

    .. math::

        r_i = \\sum_j w_{ij} r_j 

        \\frac d{dt} w_{ij} = \\eta \\cdot r_i (r_i - r_{\\theta}) r_j

    where :math:`\\eta` is some learning rate, and :math:`r_{\\theta}` is the 
    plasticity threshold,
    which is a function of the averaged postsynaptic rate, we take:

    .. math::

        r_{\\theta} = < r_i >

    **Learning Rule Parameters**
    
    ============= ============== ======== ================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------
    learning_rate 0.005          \        Learning rate.

    w_max         2.             \        Maximal possible synapse weight.

    w_min         0.             \        Minimal possible synapse weight.
    ============= ============== ======== ================================

    Returns:
        bp.Syntype: return description of the BCM rule.
        
    
    **Learning Rule State**

    ================ ================= =========================================================
    **Member name**  **Initial Value** **Explanation**
    ---------------- ----------------- ---------------------------------------------------------                                  
    w                1.                Synapse weights.
    ================ ================= =========================================================

    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """

    target_backend = 'general'

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
        dim = self.size
        r_th = bp.backend.vstack((r_th,)*dim[0])
        r_post = bp.backend.vstack((self.post.r,)*dim[0])
        r_pre = bp.backend.vstack((self.pre.r,)*dim[1]).T

        # update w
        w = self.int_w(w, _t, self.lr, r_pre, r_post, r_th)
        self.w = np.clip(w, self.w_min, self.w_max)

        # output
        self.post.r = bp.backend.sum(w.T * self.pre.r, axis=1)
