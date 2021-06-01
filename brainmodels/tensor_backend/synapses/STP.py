# -*- coding: utf-8 -*-
import brainpy as bp

__all__ = [
    'STP'
]


class STP(bp.TwoEndConn):
    """Short-term plasticity proposed by Tsodyks and Markram (Tsodyks 98) [1]_.

    The model is given by

    .. math::

        \\frac{du}{dt} = -\\frac{u}{\\tau_f}+U(1-u^-)\\delta(t-t_{spike})

        \\frac{dx}{dt} = \\frac{1-x}{\\tau_d}-u^+x^-\\delta(t-t_{spike})

    where :math:`t_{spike}` denotes the spike time and :math:`U` is the increment
    of :math:`u` produced by a spike.

    The synaptic current generated at the synapse by the spike arriving
    at :math:`t_{spike}` is then given by

    .. math::

        \\Delta I(t_{spike}) = Au^+x^-

    where :math:`A` denotes the response amplitude that would be produced
    by total release of all the neurotransmitter (:math:`u=x=1`), called
    absolute synaptic efficacy of the connections.


    **Synapse Parameters**

    ============= ============== ======== ===========================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -------------------------------------------
    tau_d         200.           ms       Time constant of short-term depression.

    tau_f         1500.          ms       Time constant of short-term facilitation.

    U             .15            \        The increment of :math:`u` produced by a spike.

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ===========================================    
    
    **Synapse Variables**

    An object of synapse class record those variables for each synapse:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    u                  0                 Release probability of the neurotransmitters.

    x                  1                 A Normalized variable denoting the fraction of remain neurotransmitters.

    w                  1                 Synapse weight.

    g                  0                 Synapse conductance.
    ================== ================= =========================================================
    

    References:
        .. [1] Tsodyks, Misha, Klaus Pawelzik, and Henry Markram. "Neural networks
                with dynamic synapses." Neural computation 10.4 (1998): 821-835.
    """

    target_backend = 'general'

    @staticmethod
    def derivative(s, u, x, t, tau, tau_d, tau_f):
        dsdt = -s / tau
        dudt = - u / tau_f
        dxdt = (1 - x) / tau_d
        return dsdt, dudt, dxdt

    def __init__(self, pre, post, conn, delay=0., U=0.15, tau_f=1500., tau_d=200., tau=8., **kwargs):
        # parameters
        self.tau_d = tau_d
        self.tau_f = tau_f
        self.tau = tau
        self.U = U
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        # variables
        self.s = bp.ops.zeros(self.size)
        self.x = bp.ops.ones(self.size)
        self.u = bp.ops.zeros(self.size)
        self.w = bp.ops.ones(self.size)
        self.I_syn = self.register_constant_delay('I_syn', size=self.size, delay_time=delay)

        self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

        super(STP, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        self.s, u, x = self.integral(self.s, self.u, self.x, _t, self.tau, self.tau_d, self.tau_f)

        pre_spike_map = bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat
        u += self.U * (1 - self.u) * pre_spike_map
        self.s += self.w * u * self.x * pre_spike_map
        x -= u * self.x * pre_spike_map

        self.u = u
        self.x = x

        self.I_syn.push(self.s)
        self.post.input += bp.ops.sum(self.I_syn.pull(), axis=0)
