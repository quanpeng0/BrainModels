# -*- coding: utf-8 -*-
import brainpy as bp

__all__ = [
    'Voltage_jump'
]


class Voltage_jump(bp.TwoEndConn):
    """Voltage jump synapses without post-synaptic neuron refractory.

    .. math::

        I_{syn} = \sum J \delta(t-t_j)


    **Synapse Variables**

    An object of synapse class record those variables for each synapse:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    s                  0.                Gating variable of the post-synaptic neuron.

    w                  0.                Synaptic weights.
    ================== ================= =========================================================

    
    """

    target_backend = 'general'

    def __init__(self, pre, post, conn, weight=1., delay=0., post_refractory=False, **kwargs):
        # parameters
        self.delay = delay
        self.post_refractory = post_refractory

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        # variables
        self.s = bp.ops.zeros(self.size)
        self.w = bp.ops.ones(self.size) * weight
        self.I_syn = self.register_constant_delay('I_syn', size=self.size, delay_time=delay)

        super(Voltage_jump, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        self.s = bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat

        self.I_syn.push(self.s * self.w)

        if self.post_refractory:
            refra_map = (1. - bp.ops.unsqueeze(self.post.refractory, 0)) * self.conn_mat
            self.post.V += bp.ops.sum(self.I_syn.pull() * refra_map, axis=0)
        else:
            self.post.V += bp.ops.sum(self.I_syn.pull(), axis=0)
