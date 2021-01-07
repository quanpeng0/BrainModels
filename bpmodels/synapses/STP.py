# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np

def get_STP(U=0.15, tau_f=1500., tau_d=200., mode = 'scalar'):
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

    @bp.integrate
    def int_u(u, t):
        return - u / tau_f

    @bp.integrate
    def int_x(x, t):
        return (1 - x) / tau_d

    ST=bp.types.SynState({'u': 0., 'x': 1., 'w': 1., 'g': 0.})

    requires = dict(
        pre=bp.types.NeuState(['spike']),
        post=bp.types.NeuState(['V', 'input'])
    )

    if mode == 'scalar':
        def update(ST, _t, pre):
            u = int_u(ST['u'], _t)
            x = int_x(ST['x'], _t)
            if pre['spike'] > 0.:
                u += U * (1-ST['u'])
                x -= u * ST['x']
            ST['u'] = np.clip(u, 0., 1.)
            ST['x'] = np.clip(x, 0., 1.)
            ST['g'] = ST['w'] * ST['u'] * ST['x']

        @bp.delayed
        def output(ST, post):
            post['input'] += ST['g']

    elif mode == 'vector':
        requires['pre2syn']=bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index')
        requires['post_slice_syn']=bp.types.Array(dim=2)

        def update(ST, _t, pre, pre2syn):
            u = int_u(ST['u'], _t)
            x = int_x(ST['x'], _t)
            for pre_id in np.where(pre['spike'] > 0.)[0]:
                syn_ids = pre2syn[pre_id]
                u_syn = u[syn_ids] + U * (1 - ST['u'][syn_ids])
                u[syn_ids] = u_syn
                x[syn_ids] -= u_syn * ST['x'][syn_ids]
            ST['u'] = np.clip(u, 0., 1.)
            ST['x'] = np.clip(x, 0., 1.)
            ST['g'] = ST['w'] * ST['u'] * ST['x']

        @bp.delayed
        def output(ST, post, post_slice_syn):
            num_post = post_slice_syn.shape[0]
            g = np.zeros(num_post, dtype=np.float_)
            for post_id in range(num_post):
                pos = post_slice_syn[post_id]
                g[post_id] = np.sum(ST['g'][pos[0]: pos[1]])
                post['input'] += g

    elif mode == 'matrix':
        requires['conn_mat']=bp.types.MatConn()

        def update(ST, _t, pre, conn_mat):
            u = int_u(ST['u'], _t)
            x = int_x(ST['x'], _t)
            spike_idxs = np.where(pre['spike'] > 0.)[0]
            #
            u_syn = u[spike_idxs] + U * (1 - ST['u'][spike_idxs])
            u[spike_idxs] = u_syn
            x[spike_idxs] -= u_syn * ST['x'][spike_idxs]
            #
            ST['u'] = np.clip(u, 0., 1.)
            ST['x'] = np.clip(x, 0., 1.)
            ST['g'] = ST['w'] * ST['u'] * ST['x']

        @bp.delayed
        def output(ST, post):
            g = np.sum(ST['g'], axis=0)
            post['input'] += g

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='STP_synapse',
                      ST=ST, requires=requires,
                      steps=(update, output),
                      mode = mode)