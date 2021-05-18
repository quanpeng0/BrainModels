# -*- coding: utf-8 -*-
import brainpy as bp

__all__ = [
    'NMDA'
]


class NMDA(bp.TwoEndConn):
    """NMDA conductance-based synapse.

    .. math::

        & I_{syn} = \\bar{g} s (V-E_{syn}) \\cdot g_{\\infty}

        & g_{\\infty}(V,[{Mg}^{2+}]_{o}) = (1+{e}^{-\\alpha V}
        \\frac{[{Mg}^{2+}]_{o}} {\\beta})^{-1} 

        & \\frac{d s_{j}(t)}{dt} = -\\frac{s_{j}(t)}
        {\\tau_{decay}}+a x_{j}(t)(1-s_{j}(t)) 

        & \\frac{d x_{j}(t)}{dt} = -\\frac{x_{j}(t)}{\\tau_{rise}}+
        \\sum_{k} \\delta(t-t_{j}^{k})


    where the decay time of NMDA currents is taken to be :math:`\\tau_{decay}` =100 ms,
    :math:`a= 0.5 ms^{-1}`, and :math:`\\tau_{rise}` =2 ms


    **Synapse Parameters**

    ============= ============== =============== ================================================
    **Parameter** **Init Value** **Unit**        **Explanation**
    ------------- -------------- --------------- ------------------------------------------------
    g_max         .15            µmho(µS)        Maximum conductance.

    E             0.             mV              The reversal potential for the synaptic current.

    alpha         .062           \               Binding constant.

    beta          3.57           \               Unbinding constant.

    cc_Mg         1.2            mM              Concentration of Magnesium ion.

    tau_decay     100.           ms              The time constant of decay.

    tau_rise      2.             ms              The time constant of rise.

    a             .5             1/ms 

    mode          'scalar'       \               Data structure of ST members.
    ============= ============== =============== ================================================    
    
    
    **Synapse Variables**

    An object of synapse class record those variables for each synapse:

	================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    s                  0                  Gating variable.
    
    g                  0                  Synapse conductance.

    x                  0                  Gating variable.
	================== ================= =========================================================

        
    References:
        .. [1] Brunel N, Wang X J. Effects of neuromodulation in a 
               cortical network model of object working memory dominated 
               by recurrent inhibition[J]. 
               Journal of computational neuroscience, 2001, 11(1): 63-85.
    
    """

    target_backend = 'general'

    @staticmethod
    def derivative(s, x, t, tau_rise, tau_decay, a):
        dxdt = -x / tau_rise
        dsdt = -s / tau_decay + a * x * (1 - s)
        return dsdt, dxdt

    def __init__(self, pre, post, conn, delay=0., g_max=0.15, E=0., cc_Mg=1.2,
                 alpha=0.062, beta=3.57, tau=100, a=0.5, tau_rise=2., **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.cc_Mg = cc_Mg
        self.tau = tau
        self.tau_rise = tau_rise
        self.a = a
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        # variables
        self.s = bp.ops.zeros(self.size)
        self.x = bp.ops.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        self.integral = bp.odeint(f=self.derivative, method='euler')

        super(NMDA, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        self.x += bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.s, self.x = self.integral(self.s, self.x, _t, self.tau_rise, self.tau, self.a)

        self.g.push(self.g_max * self.s)
        g_inf = 1 + self.cc_Mg / self.beta * bp.ops.exp(-self.alpha * self.post.V)
        g_inf = 1 / g_inf
        self.post.input -= bp.ops.sum(self.g.pull(), axis=0) * (self.post.V - self.E) * g_inf
