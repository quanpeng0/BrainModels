# -*- coding: utf-8 -*-
import brainpy as bp

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

    target_backend = 'general'

    def __init__(self, pre, post, conn, weight=1., delay=0., post_refractory=False, **kwargs):
        # parameters
        self.delay = delay
        self.post_refractory = post_refractory

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        # variables
        self.s = bp.backend.zeros(self.size)
        self.w = bp.backend.ones(self.size) * weight
        self.I_syn = self.register_constant_delay('I_syn', size=self.size, delay_time=delay)

        super(Voltage_jump, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        self.s = bp.backend.unsqueeze(self.pre.spike, 1) * self.conn_mat

        self.I_syn.push(self.s * self.w)

        if self.post_refractory:
            refra_map = (1. - bp.backend.unsqueeze(self.post.refractory, 0)) * self.conn_mat
            self.post.V += bp.backend.sum(self.I_syn.pull() * refra_map, axis=0)
        else:
            self.post.V += bp.backend.sum(self.I_syn.pull(), axis=0)
