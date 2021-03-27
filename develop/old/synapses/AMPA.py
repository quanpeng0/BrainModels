# -*- coding: utf-8 -*-

import numpy as np
from numba import prange
import brainpy as bp

bp.integrators.set_default_odeint('rk4')
bp.backend.set(backend='numba', dt=0.01)

class AMPA1(bp.TwoEndConn):
    """AMPA conductance-based synapse (type 1).

    .. math::

        I(t)&=\\bar{g} s(t) (V-E_{syn})

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})


    **Synapse Parameters**
    
    ============= ============== ======== ===================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -----------------------------------------------------------------------------------
    tau_decay     2.             ms       The time constant of decay.

    g_max         .1             µmho(µS) Maximum conductance.

    E             0.             mV       The reversal potential for the synaptic current. (only for conductance-based model)

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ===================================================================================  
    
    Returns:
        bp.Syntype: return description of the AMPA synapse model.

    **Synapse State**

    ST refers to the synapse state, items in ST are listed below:
    
    =============== ================== =========================================================
    **Member name** **Initial values** **Explanation**
    --------------- ------------------ ---------------------------------------------------------
    s                   0               Gating variable.
    
    g                   0               Synapse conductance.
    =============== ================== =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    References:
        .. [1] Brunel N, Wang X J. Effects of neuromodulation in a cortical network 
                model of object working memory dominated by recurrent inhibition[J]. 
                Journal of computational neuroscience, 2001, 11(1): 63-85.
    """

    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    def __init__(self, pre, post, conn, delay=0., g_max=0.10, E=0., tau=2.0, **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.tau = tau
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # data
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(AMPA1_vec, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def int_s(s, t, tau):
        return - s / tau

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            self.s[i] = self.int_s(self.s[i], _t, self.tau)
            self.s[i] += self.pre.spike[pre_id]
            self.g.push(i, self.g_max * self.s[i])
            post_id = self.post_ids[i]
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)

class AMPA1_mat(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel']

    def __init__(self, pre, post, conn, delay=0., g_max=0.10, E=0., tau=2.0, **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.tau = tau
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        # data
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(AMPA1_mat, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint
    def int_s(s, t, tau):
        return - s / tau

    def update(self, _t):
        self.s = self.int_s(self.s, _t, self.tau)
        for i in range(self.pre.size[0]):
            if self.pre.spike[i] > 0:
                self.s[i] += self.conn_mat[i]
        self.g.push(self.g_max * self.s)
        g = self.g.pull()
        self.post.input -= bp.backend.sum(g, axis=0) * (self.post.V - self.E)



class AMPA2(bp.TwoEndConn):
    """AMPA conductance-based synapse (type 2).

    .. math::

        I(t)&=\\bar{g} s(t) (V-E_{syn})

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})


    **Synapse Parameters**

    ============= ============== ======== ===================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -----------------------------------------------------------------------------------
    tau_decay     2.             ms       The time constant of decay.

    g_max         .1             µmho(µS) Maximum conductance.

    E             0.             mV       The reversal potential for the synaptic current. (only for conductance-based model)

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ===================================================================================

    **Synapse State**

    ST refers to the synapse state, items in ST are listed below:

    =============== ================== =========================================================
    **Member name** **Initial values** **Explanation**
    --------------- ------------------ ---------------------------------------------------------
    s                   0               Gating variable.

    g                   0               Synapse conductance.
    =============== ================== =========================================================

    Note that all ST members are saved as floating point type in BrainPy,
    though some of them represent other data types (such as boolean).

    References:
        .. [1] Brunel N, Wang X J. Effects of neuromodulation in a cortical network
                model of object working memory dominated by recurrent inhibition[J].
                Journal of computational neuroscience, 2001, 11(1): 63-85.
    """
    
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    def __init__(self, pre, post, conn, delay=0., g_max=0.42, E=0., alpha=0.98, beta=0.18, T=0.5, T_duration=0.5, **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.T_duration = T_duration
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # data
        self.s = bp.backend.zeros(self.size)
        self.t_last_pre_spike = -1e7
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(AMPA2, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def int_s(s, t, TT, alpha, beta):
        return alpha * TT * (1 - s) - beta * s

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]

            if self.pre.spike[pre_id] > 0.:
                self.t_last_pre_spike = _t

            TT = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T

            s = self.int_s(self.s[i], _t, TT, self.alpha, self.beta)
            self.s[i] = np.clip(s, 0., 1.)
            self.g.push(i, self.g_max * self.s[i])
            post_id = self.post_ids[i]
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)

