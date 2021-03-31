# -*- coding: utf-8 -*-
import brainpy as bp
from numba import prange

__all__ = [
    'GABAa1_vec',
    'GABAa1_mat',
    'GABAa2_vec',
    'GABAa2_mat',
]

class GABAa1_vec(bp.TwoEndConn):    
    """
    GABAa conductance-based synapse model.

    .. math::

        I_{syn}&= - \\bar{g}_{max} s (V - E)

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k}\\delta(t-t-{j}^{k})


    **Synapse Parameters**

    ============= ============== ======== =======================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ---------------------------------------
    g_max         0.4            \        Maximum synapse conductance.

    E             -80.           \        Reversal potential of synapse.

    tau_decay     6.             ms       Time constant of gating variable decay.
    ============= ============== ======== =======================================

    **Synapse Variables**    

    An object of synapse class record those variables for each synapse:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    s                  0.                Gating variable.

    g                  0.                Synapse conductance on post-synaptic neuron.
    ================== ================= =========================================================

    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numba-cuda']

    def __init__(self, pre, post, conn, delay=0., 
                 g_max=0.4, E=-80., tau_decay=6., 
                 **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.tau_decay = tau_decay
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # data
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(GABAa1_vec, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def integral(s, t, tau_decay):
        return - s / tau_decay

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            self.s[i] = self.integral(self.s[i], _t, self.tau_decay)
            self.s[i] += self.pre.spike[pre_id]
            #pdb.set_trace()
            self.g.push(i,self.g_max * self.s[i])
            post_id = self.post_ids[i]
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)


class GABAa1_mat(bp.TwoEndConn):
    """
    GABAa conductance-based synapse model.

    .. math::

        I_{syn}&= - \\bar{g}_{max} s (V - E)

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k}\\delta(t-t-{j}^{k})


    **Synapse Parameters**

    ============= ============== ======== =======================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ---------------------------------------
    g_max         0.4            \        Maximum synapse conductance.

    E             -80.           \        Reversal potential of synapse.

    tau_decay     6.             ms       Time constant of gating variable decay.
    ============= ============== ======== =======================================

    **Synapse Variables**    

    An object of synapse class record those variables for each synapse:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    s                  0.                Gating variable.

    g                  0.                Synapse conductance on post-synaptic neuron.
    ================== ================= =========================================================

    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """
    target_backend = ['numpy', 'numba', 'numba-parallel']

    def __init__(self, pre, post, conn, delay=0., 
                 g_max=0.4, E=-80., tau_decay=6., 
                 **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.tau_decay = tau_decay
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        # data
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(GABAa1_mat, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint
    def int_s(s, t, tau_decay):
        return - s / tau_decay

    def update(self, _t):
        self.s = self.int_s(self.s, _t, self.tau_decay)
        for i in range(self.pre.size[0]):
            if self.pre.spike[i] > 0:
                self.s[i] += self.conn_mat[i]
        self.g.push(self.g_max * self.s)
        g=self.g.pull()
        self.post.input -= bp.backend.sum(g, axis=0) * (self.post.V - self.E)


class GABAa2_vec(bp.TwoEndConn):
    target_backend = 'general'

    def __init__(self, pre, post, conn, delay = 0.,
                 g_max=0.04, E=-80., alpha=0.53, 
                 beta=0.18, T=1., T_duration=1.,
                 **kwargs):
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.T_duration = T_duration

        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = self.conn.requires(
            'pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size = self.size, delay_time = delay)
        self.t_last_pre_spike = bp.backend.ones(self.size) * -1e7
        super(GABAa2_vec, self).__init__(pre = pre, post = post, **kwargs)

    @staticmethod
    @bp.odeint
    def integral(s, t, TT, alpha, beta):
        dsdt = alpha * TT * (1 - s) - beta * s
        return dsdt

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            post_id = self.post_ids[i]
            if self.pre.spike[pre_id]:
                self.t_last_pre_spike[i] = _t
            T = ((_t - self.t_last_pre_spike[i]) < self.T_duration) * self.T
            self.s[i] = self.integral(self.s[i], _t, T, self.alpha, self.beta)
            self.g.push(i, self.s[i] * self.g_max)
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)
                

class GABAa2_mat(bp.TwoEndConn):
    target_backend = 'general'

    def __init__(self, pre, post, conn, delay = 0.,
                 g_max=0.04, E=-80., alpha=0.53, 
                 beta=0.18, T=1., T_duration=1.,
                 **kwargs):
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.T_duration = T_duration

        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size = self.size, delay_time = delay)
        self.t_last_pre_spike = bp.backend.ones(self.size) * -1e7
        super(GABAa2_mat, self).__init__(pre = pre, post = post, **kwargs)

    @staticmethod
    @bp.odeint
    def integral(s, t, TT, alpha, beta):
        dsdt = alpha * TT * (1 - s) - beta * s
        return dsdt

    def update(self, _t):
        spike = bp.backend.reshape(self.pre.spike, (-1, 1)) * self.conn_mat
        self.t_last_pre_spike = bp.backend.where(spike, _t, self.t_last_pre_spike)
        TT = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
        self.s = self.integral(self.s, _t, TT, self.alpha, self.beta)
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.backend.sum(self.g.pull(), 0) * (self.post.V - self.E)

