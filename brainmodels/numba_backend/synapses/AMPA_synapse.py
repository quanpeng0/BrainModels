# -*- coding: utf-8 -*-

import brainpy as bp
from numba import prange

__all__ = [
    'AMPA1',
    'AMPA2',
]


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

    @staticmethod
    def derivative(s, t, tau):
        ds = - s / tau
        return ds

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

        self.int_s = bp.odeint(f=self.derivative, method='exponential_euler')
        super(AMPA1, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            self.s[i] = self.int_s(self.s[i], _t, self.tau)
            self.s[i] += self.pre.spike[pre_id]
            self.g.push(i, self.g_max * self.s[i])
            post_id = self.post_ids[i]
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)


class AMPA2(bp.TwoEndConn):
    """AMPA conductance-based synapse (type 2).

    """

    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    @staticmethod
    def derivative(s, t, TT, alpha, beta):
        ds = alpha * TT * (1 - s) - beta * s
        return ds

    def __init__(self, pre, post, conn, delay=0., g_max=0.42, E=0.,
                 alpha=0.98, beta=0.18, T=0.5, T_duration=0.5, **kwargs):
        # parameters
        self.delay = delay
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.T_duration = T_duration

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # variables
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)
        self.t_last_pre_spike = -1e7 * bp.backend.ones(self.size)

        self.int_s = bp.odeint(f=self.derivative, method='exponential_euler')
        super(AMPA2, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            post_id = self.post_ids[i]

            if self.pre.spike[pre_id]:
                self.t_last_pre_spike[pre_id] = _t
            TT = ((_t - self.t_last_pre_spike[pre_id]) < self.T_duration) * self.T
            self.s[i] = self.int_s(self.s[i], _t, TT, self.alpha, self.beta)
            self.g.push(i, self.g_max * self.s[i])
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)
