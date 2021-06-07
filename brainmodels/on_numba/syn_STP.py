# -*- coding: utf-8 -*-
import brainpy as bp
from numba import prange

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
    
    Returns:
        bp.Syntype: return description of the Short-term plasticity synapse model.

    **Synapse State**

    ST refers to the synapse state, items in ST are listed below:
    
    =============== ================== =====================================================================
    **Member name** **Initial values** **Explanation**
    --------------- ------------------ ---------------------------------------------------------------------
    u                 0                 Release probability of the neurotransmitters.

    x                 1                 A Normalized variable denoting the fraction of remain neurotransmitters.

    w                 1                 Synapse weight.

    g                 0                 Synapse conductance.
    =============== ================== =====================================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).


    References:
        .. [1] Tsodyks, Misha, Klaus Pawelzik, and Henry Markram. "Neural networks
                with dynamic synapses." Neural computation 10.4 (1998): 821-835.
    """

    target_backend = ['numpy', 'numba', 'numba-parallel', 'numba-cuda']

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
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # variables
        self.s = bp.ops.zeros(self.size)
        self.x = bp.ops.ones(self.size)
        self.u = bp.ops.zeros(self.size)
        self.w = bp.ops.ones(self.size)
        self.I_syn = self.register_constant_delay('I_syn', size=self.size, delay_time=delay)

        self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

        super(STP, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]

            self.s[i], u, x = self.integral(self.s[i], self.u[i], self.x[i], _t, self.tau, self.tau_d, self.tau_f)

            if self.pre.spike[pre_id] > 0:
                u += self.U * (1 - self.u[i])
                self.s[i] += self.w[i] * u * self.x[i]
                x -= u * self.x[i]
            self.u[i] = u
            self.x[i] = x

            # output
            post_id = self.post_ids[i]
            self.I_syn.push(i, self.s[i])
            self.post.input[post_id] += self.I_syn.pull(i)
