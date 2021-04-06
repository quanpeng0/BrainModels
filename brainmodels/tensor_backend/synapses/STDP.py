# -*- coding: utf-8 -*-
import brainpy as bp

__all__ = [
    'STDP'
]


class STDP(bp.TwoEndConn):
    """
    Spike-time dependent plasticity.

    .. math::

        \\frac{d A_{source}}{d t}&=-\\frac{A_{source}}{\\tau_{source}}

        \\frac{d A_{target}}{d t}&=-\\frac{A_{target}}{\\tau_{target}}

    After a pre-synaptic spike:

    .. math::      

        s_{post}&= s_{post}+w

        A_{source}&= A_{source} + \\Delta A_{source}

        w&= min([w-A_{target}]^+, w_{max})

    After a post-synaptic spike:

    .. math::

        A_{target}&= A_{target} + \\Delta A_{target}

        w&= min([w+A_{source}]^+, w_{max})

    **Learning Rule Parameters**

    ============= ============== ======== =================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -----------------------------------------------------------------
    tau           10.            ms       Time constant of decay.

    tau_s         10.            ms       Time constant of source neuron 

                                          (i.e. pre-synaptic neuron)

    tau_t         10.            ms       Time constant of target neuron 

                                          (i.e. post-synaptic neuron)

    w_min         0.             \        Minimal possible synapse weight.

    w_max         20.            \        Maximal possible synapse weight.

    delta_A_s     0.5            \        Change on source neuron traces elicited by 

                                          a source neuron spike.

    delta_A_t     0.5            \        Change on target neuron traces elicited by 

                                          a target neuron spike.

    ============= ============== ======== =================================================================

    Returns:
        bp.Syntype: return description of STDP.


    **Learning Rule State**

    ST refers to synapse state (note that STDP learning rule can be implemented as synapses),
    members of ST are listed below:

    ================ ================= =========================================================
    **Member name**  **Initial Value** **Explanation**
    ---------------- ----------------- ---------------------------------------------------------
    A_s              0.                Source neuron trace.

    A_t              0.                Target neuron trace.

    s                0.                Gating variable on post-synaptic neuron.

    w                0.                Synapse weight.
    ================ ================= =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    References:
        .. [1] Stimberg, Marcel, et al. "Equation-oriented specification of neural models for
               simulations." Frontiers in neuroinformatics 8 (2014): 6.
    """

    target_backend = 'general'

    @staticmethod
    def derivative(s, A_s, A_t, t, tau, tau_s, tau_t):
        dsdt = -s / tau
        dAsdt = - A_s / tau_s
        dAtdt = - A_t / tau_t
        return dsdt, dAsdt, dAtdt

    def __init__(self, pre, post, conn, delay=0.,
                 delta_A_s=0.5, delta_A_t=0.5, w_min=0., w_max=20.,
                 tau_s=10., tau_t=10., tau=10., **kwargs):
        # parameters
        self.tau_s = tau_s
        self.tau_t = tau_t
        self.tau = tau
        self.delta_A_s = delta_A_s
        self.delta_A_t = delta_A_t
        self.w_min = w_min
        self.w_max = w_max
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        # variables
        self.s = bp.backend.zeros(self.size)
        self.A_s = bp.backend.zeros(self.size)
        self.A_t = bp.backend.zeros(self.size)
        self.w = bp.backend.ones(self.size) * 1.
        self.out = self.register_constant_delay('out', size=self.size, delay_time=delay)

        self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

        super(STDP, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        s, A_s, A_t = self.integral(self.s, self.A_s, self.A_t,
                                    _t, self.tau, self.tau_s, self.tau_t)
        w = self.w

        pre_spike_map = bp.backend.unsqueeze(self.pre.spike, 1) * self.conn_mat
        s += w * pre_spike_map
        A_s += self.delta_A_s * pre_spike_map
        w -= A_t * pre_spike_map

        post_spike_map = bp.backend.unsqueeze(self.post.spike, 0) * self.conn_mat
        A_t += self.delta_A_t * post_spike_map
        w += A_s * post_spike_map

        self.A_s = A_s
        self.A_t = A_t
        self.w = bp.backend.clip(w, self.w_min, self.w_max)
        self.s = s

        self.out.push(self.s)
        self.post.input += bp.backend.sum(self.out.pull(), axis=0)
