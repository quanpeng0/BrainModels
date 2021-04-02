# -*- coding: utf-8 -*-

import brainpy as bp

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

    target_backend = 'general'

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
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        # data
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        self.int_s = bp.odeint(f=self.derivative, method='euler')
        super(AMPA1, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        self.s = self.int_s(self.s, _t, self.tau)
        self.s += bp.backend.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.backend.sum(self.g.pull(), 0) * (self.post.V - self.E)


class AMPA2(bp.TwoEndConn):
    """AMPA conductance-based synapse (type 2).
    
    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{ds}{dt} &=\\alpha[T](1-s)-\\beta s

    **Synapse Parameters**
    
    ============= ============== ======== ================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------
    g_max         .42            µmho(µS) Maximum conductance.

    E             0.             mV       The reversal potential for the synaptic current.

    alpha         .98            \        Binding constant.

    beta          .18            \        Unbinding constant.

    T             .5             mM       Neurotransmitter concentration.

    T_duration    .5             ms       Duration of the neurotransmitter concentration.

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ================================================    
    
    Returns:
        bp.Syntype: return description of the AMPA synapse model.

    **Synapse State**

    ST refers to the synapse state, items in ST are listed below:
    
    ================ ================== =========================================================
    **Member name**  **Initial values** **Explanation**
    ---------------- ------------------ ---------------------------------------------------------
    s                 0                 Gating variable.
    
    g                 0                 Synapse conductance.

    t_last_pre_spike  -1e7              Last spike time stamp of the pre-synaptic neuron.
    ================ ================== =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    References:
        .. [1] Vijayan S, Kopell N J. Thalamic model of awake alpha oscillations 
                and implications for stimulus processing[J]. Proceedings of the 
                National Academy of Sciences, 2012, 109(45): 18553-18558.
    """

    target_backend = 'general'

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
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        # variables
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)
        self.t_last_pre_spike = -1e7 * bp.backend.ones(self.size)

        self.int_s = bp.odeint(f=self.derivative, method='euler')
        super(AMPA2, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        spike = bp.backend.reshape(self.pre.spike, (-1, 1)) * self.conn_mat
        self.t_last_pre_spike = bp.backend.where(spike, _t, self.t_last_pre_spike)
        TT = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
        self.s = self.int_s(self.s, _t, TT, self.alpha, self.beta)
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.backend.sum(self.g.pull(), 0) * (self.post.V - self.E)


class LIF2(bp.NeuGroup):
    target_backend = 'general'

    @staticmethod
    def derivative(V, t, Iext, V_rest, R, tau):
        return (-V + V_rest + R * Iext) / tau

    def __init__(self, size, t_refractory=1., V_rest=0.,
                 V_reset=-5., V_th=20., R=1., tau=10., **kwargs):
        # parameters
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.t_refractory = t_refractory

        # variables
        num = bp.size2len(size)
        self.t_last_spike = bp.backend.ones(num) * -1e7
        self.input = bp.backend.zeros(num)
        self.V = bp.backend.ones(num) * V_reset
        self.refractory = bp.backend.zeros(num, dtype=bool)
        self.spike = bp.backend.zeros(num, dtype=bool)

        self.int_V = bp.odeint(self.derivative)
        super(LIF2, self).__init__(size=size, **kwargs)

    def update(self, _t):
        refractory = (_t - self.t_last_spike) <= self.t_refractory
        V = self.int_V(self.V, _t, self.input, self.V_rest, self.R, self.tau)
        V = bp.backend.where(refractory, self.V, V)
        spike = V >= self.V_th
        self.t_last_spike = bp.backend.where(spike, _t, self.t_last_spike)
        self.V = bp.backend.where(spike, self.V_reset, V)
        self.refractory = refractory
        self.input[:] = 0.
        self.spike = spike


import matplotlib.pyplot as plt

duration = 100.
dt = 0.02

size = 10
neu_pre = LIF2(size, monitors=['V', 'input', 'spike'], )
neu_pre.V_rest = -65.
neu_pre.V_reset = -70.
neu_pre.V_th = -50.
neu_pre.V = bp.backend.ones(size) * -65.
neu_post = LIF2(size, monitors=['V', 'input', 'spike'], )

syn_GABAb = AMPA1(pre=neu_pre, post=neu_post, conn=bp.connect.One2One(),
                   delay=10., monitors=['s'], )

net = bp.Network(neu_pre, syn_GABAb, neu_post)
net.run(200, inputs=[(neu_pre, 'input', 25)], report=True)

print(neu_pre.mon.spike.max())
print(syn_GABAb.mon.s.max())

bp.visualize.line_plot(net.ts, neu_pre.mon.V, show=True)

# paint gabaa
ts = net.ts
# print(gabaa.mon.s.shape)
plt.plot(ts, syn_GABAb.mon.s[:, 0, 0], label='s')
plt.legend()
plt.show()

