# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
from numba import prange

# bp.integrators.set_default_odeint('rk4')
bp.backend.set(backend='numba', dt=0.01)

class Gap_junction(bp.TwoEndConn):
    """
    synapse with gap junction.

    .. math::

        I_{syn} = w (V_{pre} - V_{post})

    **Synapse State**

    ST refers to synapse state, members of ST are listed below:

    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    w                0.                Synapse weights.
    =============== ================= =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        mode (string): data structure of ST members.

    Returns:
        bp.SynType: return description of synapse model with gap junction.

    Reference:
        .. [1] Chow, Carson C., and Nancy Kopell. 
                "Dynamics of spiking neurons with electrical coupling." 
                Neural computation 12.7 (2000): 1643-1678.

    """

    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    def __init__(self, pre, post, conn, delay=0., **kwargs):
        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)
        self.delay = delay

        # variables
        self.w = bp.backend.ones(self.size)

        super(Gap_junction, self).__init__(pre=pre, post=post, **kwargs)


    
    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            post_id = self.post_ids[i]

            self.post.input[post_id] += self.w[i] * (self.pre.V[pre_id] - self.post.V[post_id])



class Gap_junction_lif(bp.TwoEndConn):
    """
    synapse with gap junction.

    .. math::

        I_{syn} = w (V_{pre} - V_{post})

    ST refers to synapse state, members of ST are listed below:

    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    w                0.                Synapse weights.
    
    spikelet         0.                conductance for post-synaptic neuron
    =============== ================= =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        k_spikelet (float): 

    Returns:
        bp.SynType: return description of synapse model with gap junction.

    References:
        .. [1] Chow, Carson C., and Nancy Kopell. 
                "Dynamics of spiking neurons with electrical coupling." 
                Neural computation 12.7 (2000): 1643-1678.

    """
    
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    def __init__(self, pre, post, conn, delay=0., k_spikelet=0.1, post_refractory=False,  **kwargs):
        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)
        self.delay = delay
        self.k_spikelet = k_spikelet
        self.post_refractory = post_refractory

        # variables
        self.w = bp.backend.ones(self.size)
        self.spikelet = self.register_constant_delay('spikelet', size=self.size, delay_time=delay)

        super(Gap_junction_lif, self).__init__(pre=pre, post=post, **kwargs)


    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            post_id = self.post_ids[i]

            self.post.input[post_id] += self.w[i] * (self.pre.V[pre_id] - self.post.V[post_id])

            if self.post_refractory:
                self.spikelet.push(i, self.w[i] * self.k_spikelet * self.pre.spike[pre_id] * (1. - self.post.refractory[post_id]))
            else:
                self.spikelet.push(i, self.w[i] * self.k_spikelet * self.pre.spike[pre_id])
            
            self.post.V[post_id] += self.spikelet.pull(i)
