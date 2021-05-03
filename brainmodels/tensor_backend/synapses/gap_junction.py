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

    **Synapse Variables**

    An object of synapse class record those variables for each synapse:

	================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    w                0.                Synapse weights.
    =============== ================= =========================================================

    Reference:
        .. [1] Chow, Carson C., and Nancy Kopell. 
                "Dynamics of spiking neurons with electrical coupling." 
                Neural computation 12.7 (2000): 1643-1678.

    """

    target_backend = 'general'

    def __init__(self, pre, post, conn, **kwargs):
        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        # variables
        self.w = bp.ops.ones(self.size)

        super(Gap_junction, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        v_post = bp.ops.vstack((self.post.V,) * self.size[0])
        v_pre = bp.ops.vstack((self.pre.V,) * self.size[1]).T

        I_syn = self.w * (v_pre - v_post) * self.conn_mat
        self.post.input += bp.ops.sum(I_syn, axis=0)


class Gap_junction_lif(bp.TwoEndConn):
    """
    synapse with gap junction.

    .. math::

        I_{syn} = w (V_{pre} - V_{post})

    **Synapse Variables**

    An object of synapse class record those variables for each synapse:

	================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    w                0.                Synapse weights.
    
    spikelet         0.                conductance for post-synaptic neuron
    =============== ================= =========================================================

    References:
        .. [1] Chow, Carson C., and Nancy Kopell. 
                "Dynamics of spiking neurons with electrical coupling." 
                Neural computation 12.7 (2000): 1643-1678.

    """

    target_backend = 'general'

    def __init__(self, pre, post, conn, delay=0., k_spikelet=0.1, post_refractory=False, **kwargs):
        self.delay = delay
        self.k_spikelet = k_spikelet
        self.post_refractory = post_refractory

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        # variables
        self.w = bp.ops.ones(self.size)
        self.spikelet = self.register_constant_delay('spikelet', size=self.size, delay_time=self.delay)

        super(Gap_junction_lif, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        v_post = bp.ops.vstack((self.post.V,) * self.size[0])
        v_pre = bp.ops.vstack((self.pre.V,) * self.size[1]).T

        I_syn = self.w * (v_pre - v_post) * self.conn_mat
        self.post.input += bp.ops.sum(I_syn, axis=0)

        self.spikelet.push(self.w * self.k_spikelet * bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat)

        if self.post_refractory:
            self.post.V += bp.ops.sum(self.spikelet.pull(), axis=0) * (1. - self.post.refractory)
        else:
            self.post.V += bp.ops.sum(self.spikelet.pull(), axis=0)
