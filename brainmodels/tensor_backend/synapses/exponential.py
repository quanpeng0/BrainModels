# -*- coding: utf-8 -*-
import brainpy as bp

__all__ = [
    'Exponential'
]
class Exponential(bp.TwoEndConn):
    '''
    Single Exponential decay synapse model.

    .. math::

         \\frac{d s}{d t} = -\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})

    For conductance-based (co-base=True):

    .. math::
    
        I_{syn}(t) = g_{syn} (t) (V(t)-E_{syn})


    For current-based (co-base=False):

    .. math::
    
        I(t) = \\bar{g} s (t)


    **Synapse Parameters**
    
    ============= ============== ======== ===================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -----------------------------------------------------------------------------------
    tau_decay     8.             ms       The time constant of decay.

    ============= ============== ======== ===================================================================================  
    
    Returns:
        bp.Syntype: return description of a synapse model with single exponential decay.

    **Synapse State**
    
    ================ ================== =========================================================
    **Member name**  **Initial values** **Explanation**
    ---------------- ------------------ ---------------------------------------------------------    
    s                  0                  Gating variable.

    w                  0                  Synaptic weights.
                             
    ================ ================== =========================================================

    References:
        .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw. 
                "The Synapse." Principles of Computational Modelling in Neuroscience. 
                Cambridge: Cambridge UP, 2011. 172-95. Print.
    '''

    target_backend = 'general'

    @staticmethod
    def derivative(s, t, tau):
        return -s / tau

    def __init__(self, pre, post, conn, delay=0., tau=8.0, **kwargs):
        # parameters
        self.tau = tau
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        # variables
        self.s = bp.backend.zeros(self.size)
        self.w = bp.backend.ones(self.size) * .1
        self.out = self.register_constant_delay('out', size=self.size, delay_time=delay)

        self.integral = bp.odeint(f=self.derivative, method='euler')

        super(Exponential, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        self.s = self.integral(self.s, _t, self.tau)
        self.s += bp.backend.unsqueeze(self.post.spike, 1) * self.conn_mat
        self.out.push(self.w * self.s)
        self.post.input += bp.backend.sum(self.out.pull(), axis=0)