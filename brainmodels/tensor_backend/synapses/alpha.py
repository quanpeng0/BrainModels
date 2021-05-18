# -*- coding: utf-8 -*-
import brainpy as bp

__all__ = [
    'Alpha'
]


class Alpha(bp.TwoEndConn):
    """
    Alpha synapse.

    .. math::
        \\frac {ds} {dt} &= x

        \\tau^2 \\frac {dx} {dt} = - 2 \\tau x & - s + \\sum_f \\delta(t-t^f)


    **Synapse Parameters**

    ============= ============== ======== ===================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -----------------------------------------------------------------------------------
    tau_decay     2.             ms       The time constant of decay.

    g_max         .2             µmho(µS) Maximum conductance.

    E             0.             mV       The reversal potential for the synaptic current. (only for conductance-based model)

    co_base       False          \        Whether to return Conductance-based model. If False: return current-based model.

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ===================================================================================  


    **Synapse Variables**

    An object of synapse class record those variables for each synapse:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    g                  0                  Synapse conductance on the post-synaptic neuron.
    s                  0                  Gating variable.
    x                  0                  Gating variable.  
    ================== ================= =========================================================

    References:
        .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw. 
                "The Synapse." Principles of Computational Modelling in Neuroscience. 
                Cambridge: Cambridge UP, 2011. 172-95. Print.
    """

    target_backend = 'general'

    @staticmethod
    def derivative(s, x, t, tau):
        dxdt = (-2 * tau * x - s) / (tau ** 2)
        dsdt = x
        return dsdt, dxdt

    def __init__(self, pre, post, conn, delay=0., tau=2.0, **kwargs):
        # parameters
        self.tau = tau
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        # variables
        self.s = bp.ops.zeros(self.size)
        self.x = bp.ops.zeros(self.size)

        self.w = bp.ops.ones(self.size) * .2
        self.I_syn = self.register_constant_delay('I_syn', size=self.size, delay_time=delay)

        self.integral = bp.odeint(f=self.derivative, method='euler')
        super(Alpha, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        self.s, self.x = self.integral(self.s, self.x, _t, self.tau)
        self.x += bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.I_syn.push(self.w * self.s)
        self.post.input += bp.ops.sum(self.I_syn.pull(), axis=0)
