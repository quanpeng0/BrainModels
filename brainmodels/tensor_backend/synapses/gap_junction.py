# -*- coding: utf-8 -*-
import brainpy as bp

__all__ = [
    'Gap_junction',
    'Gap_junction_lif',
]

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

    target_backend = 'general'

    def __init__(self, pre, post, conn, delay=0., **kwargs):
        self.delay = delay
        
        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        # variables
        self.w = bp.backend.ones(self.size)

        super(Gap_junction, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        v_post = bp.backend.vstack((self.post.V,) * self.size[0])
        v_pre = bp.backend.vstack((self.pre.V,) * self.size[1]).T

        out = self.w * (v_pre - v_post) * self.conn_mat
        self.post.input += bp.backend.sum(out, axis=0)



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
    
    target_backend = 'general'

    def __init__(self, pre, post, conn, delay=0., k_spikelet=0.1, post_refractory=False,  **kwargs):
        self.delay = delay
        self.k_spikelet = k_spikelet
        self.post_refractory = post_refractory

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        # variables
        self.w = bp.backend.ones(self.size)
        self.spikelet = self.register_constant_delay('spikelet', size=self.size, delay_time=delay)

        super(Gap_junction_lif, self).__init__(pre=pre, post=post, **kwargs)


    def update(self, _t):
        v_post = bp.backend.vstack((self.post.V,) * self.size[0])
        v_pre = bp.backend.vstack((self.pre.V,) * self.size[1]).T

        out = self.w * (v_pre - v_post) * self.conn_mat
        self.post.input += bp.backend.sum(out, axis=0)

        if self.post_refractory:
            self.spikelet.push(self.w * self.k_spikelet * bp.backend.unsqueeze(self.pre.spike, 1) * self.conn_mat * (1. - self.post.refractory))
        else:
            self.spikelet.push(self.w * self.k_spikelet * bp.backend.unsqueeze(self.pre.spike, 1) * self.conn_mat)
        
        self.post.V += bp.backend.sum(self.spikelet.pull(), axis=0)
