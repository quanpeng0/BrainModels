# -*- coding: utf-8 -*-
import brainpy as bp
from numba import prange

__all__ = [
    'Voltage_jump'
]
class Voltage_jump(bp.TwoEndConn):
    """Voltage jump synapses without post-synaptic neuron refractory.

    .. math::

        I_{syn} = \sum J \delta(t-t_j)


    ST refers to synapse state, members of ST are listed below:
    
    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    s               0.                Gating variable of the post-synaptic neuron.
    =============== ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        post_has_refractory (bool): whether the post-synaptic neuron have refractory.

    Returns:
        bp.SynType.
    
    """
        
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numba-cuda']

    def __init__(self, pre, post, conn, delay=0., post_refractory=False,  **kwargs):
        # parameters
        self.delay = delay
        self.post_refractory = post_refractory

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # variables
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(Voltage_jump, self).__init__(pre=pre, post=post, **kwargs)

    
    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            self.s[i] = self.pre.spike[pre_id]

            # output
            post_id = self.post_ids[i]
            if self.post_refractory:
                self.g.push(i, self.s[i] * (1. - self.post.refractory[post_id]))
            else:
                self.g.push(i, self.s[i])
            
            self.post.V[post_id] += self.g.pull(i)