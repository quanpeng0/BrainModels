# -*- coding: utf-8 -*-
import brainpy as bp
from numba import prange

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

    target_backend = ['numpy', 'numba', 'numba-parallel', 'numba-cuda']

    @staticmethod
    def derivative(s, t, tau):
        return -s / tau

    def __init__(self, pre, post, conn, delay=0., tau=8.0, **kwargs):
        # parameters
        self.tau = tau
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # variables
        self.s = bp.backend.zeros(self.size)
        self.w = bp.backend.ones(self.size) * .1
        self.out = self.register_constant_delay('out', size=self.size, delay_time=delay)

        self.integral = bp.odeint(f=self.derivative, method='euler')

        super(Exponential, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]

            self.s[i] = self.integral(self.s[i], _t, self.tau)
            self.s[i] += self.pre.spike[pre_id]
            self.out.push(i, self.w[i] * self.s[i])

            # output
            post_id = self.post_ids[i]
            self.post.input[post_id] += self.out.pull(i) 