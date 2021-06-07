# -*- coding: utf-8 -*-
import brainpy as bp
from numba import prange

__all__ = [
    'GABAa1',
    'GABAa2',
]

class GABAa1(bp.TwoEndConn):    
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
    target_backend = ['numba', 'numba-parallel', 'numba-cuda']

    @staticmethod
    def derivative(s, t, tau_decay):
        dsdt = - s / tau_decay
        return dsdt

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
        self.s = bp.ops.zeros(self.size)
        self.g = self.register_constant_delay(
            'g', size=self.size, delay_time=delay
        )

        self.integral = bp.odeint(
            f=self.derivative, method='exponential_euler'
        )
        super(GABAa1, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            self.s[i] = self.integral(self.s[i], _t, self.tau_decay)
            self.s[i] += self.pre.spike[pre_id]
            self.g.push(i, self.g_max * self.s[i])
            post_id = self.post_ids[i]
            self.post.input[post_id] -= self.g.pull(i) \
                                        * (self.post.V[post_id] - self.E)
            
            
class GABAa2(bp.TwoEndConn):
    """
    GABAa conductance-based synapse model (markov form).

    .. math::

        I_{syn}&= - \\bar{g}_{max} s (V - E)

        \\frac{d s}{d t}&=\\alpha[T](1-s) - \\beta s

    **Synapse Parameters**

    ============= ============== ======== =======================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ---------------------------------------
    g_max         0.04           \        Maximum synapse conductance.

    E             -80.           \        Reversal potential of synapse.

    alpha         0.53           \        Activating rate constant of G protein catalyzed

                                          by activated GABAb receptor.

    beta          0.18           \        De-activating rate constant of G protein.

    T             1.             \        Transmitter concentration when synapse is

                                          triggered by a pre-synaptic spike.

    T_duration    1.             \        Transmitter concentration duration time

                                          after being triggered.
    ============= ============== ======== =======================================

    **Synapse Variables**

    An object of synapse class record those variables for each synapse:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    s                  0.                Gating variable.

    g                  0.                Synapse conductance on post-synaptic neuron.

    t_last_pre_spike   -1e7              Last spike time stamp of pre-synaptic neuron.
    ================== ================= =========================================================

    References:
        .. [1] Destexhe, Alain, and Denis Paré. "Impact of network activity
               on the integrative properties of neocortical pyramidal neurons
               in vivo." Journal of neurophysiology 81.4 (1999): 1531-1547.
    """
    target_backend = ['numba', 'numba-parallel', 'numba-cuda']

    @staticmethod
    def derivative(s, t, TT, alpha, beta):
        dsdt = alpha * TT * (1 - s) - beta * s
        return dsdt

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

        self.s = bp.ops.zeros(self.size)
        self.g = self.register_constant_delay(
            'g', size = self.size, delay_time = delay
        )
        self.t_last_pre_spike = bp.ops.ones(self.size) * -1e7

        self.integral = bp.odeint(
            f=self.derivative, method='exponential_euler'
        )
        super(GABAa2, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            post_id = self.post_ids[i]
            if self.pre.spike[pre_id]:
                self.t_last_pre_spike[i] = _t
            T = ((_t - self.t_last_pre_spike[i]) < self.T_duration) * self.T
            self.s[i] = self.integral(self.s[i], _t, T, self.alpha, self.beta)
            self.g.push(i, self.s[i] * self.g_max)
            self.post.input[post_id] -= self.g.pull(i) \
                                        * (self.post.V[post_id] - self.E)
