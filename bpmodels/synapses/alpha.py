# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np


def get_alpha(tau_decay=2., g_max=.2, E=0.,
              mode='scalar', co_base=False):
    """
    Alpha synapse.

    .. math::

        \\frac {ds} {dt} &= x

        \\tau^2 \\frac {dx} {dt} = - 2 \\tau x & - s + \\sum_f \\delta(t-t^f)


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
    tau_decay     2.             ms       The time constant of decay.

    g_max         .2             µmho(µS) Maximum conductance.

    E             0.             mV       The reversal potential for the synaptic current. (only for conductance-based model)

    co_base       False          \        Whether to return Conductance-based model. If False: return current-based model.

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ===================================================================================  

    Returns:
        bp.Syntype: return description of the alpha synapse model.

    **Synapse State**


    ST refers to the synapse state, items in ST are listed below:

    ================ ================== =========================================================
    **Member name**  **Initial values** **Explanation**
    ---------------- ------------------ ---------------------------------------------------------    
    g                  0                  Synapse conductance on the post-synaptic neuron.
    s                  0                  Gating variable.
    x                  0                  Gating variable.  
    ================ ================== =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    References:
        .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw. 
                "The Synapse." Principles of Computational Modelling in Neuroscience. 
                Cambridge: Cambridge UP, 2011. 172-95. Print.
    """

    ST = bp.types.SynState('g', 's', 'x')

    requires = {
        'pre': bp.types.NeuState('spike'),
        'post': bp.types.NeuState('input', 'V')
    }

    @bp.integrate
    def int_s(s, t, x):
        return x

    @bp.integrate
    def int_x(x, t, s):
        return (-2 * tau_decay * x - s) / (tau_decay ** 2)

    if mode == 'scalar':
        def update(ST, _t, pre):
            s = ST['s']
            x = ST['x']
            x = int_x(x, _t, s)
            s = int_s(s, _t, x)
            if pre['spike'] > 0.:
                x += 1
            ST['x'] = x
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post):
            if co_base:
                post['input'] += ST['g'] * (post['V'] - E)
            else:
                post['input'] += ST['g']

    elif mode == 'vector':
        requires['pre2syn'] = bp.types.ListConn()
        requires['post_slice_syn'] = bp.types.Array(dim=2)

        def update(ST, _t, pre, pre2syn):
            s = ST['s']
            x = ST['x']
            x = int_x(x, _t, s)
            s = int_s(s, _t, x)
            for i in np.where(pre['spike'] > 0.)[0]:
                syn_ids = pre2syn[i]
                x[syn_ids] += 1.
            ST['x'] = x
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post, post_slice_syn):
            num_post = post_slice_syn.shape[0]
            g = np.zeros(num_post, dtype=np.float_)
            for post_id in range(num_post):
                pos = post_slice_syn[post_id]
                g[post_id] = np.sum(ST['g'][pos[0]: pos[1]])
            if co_base:
                post['input'] += g * (post['V'] - E)
            else:
                post['input'] += g

    elif mode == 'matrix':
        requires['conn_mat'] = bp.types.MatConn()

        def update(ST, _t, pre, conn_mat):
            s = ST['s']
            x = ST['x']
            x = int_x(x, _t, s)
            s = int_s(s, _t, x)
            x += pre['spike'].reshape((-1, 1)) * conn_mat
            ST['x'] = x
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post):
            g = np.sum(ST['g'], axis=0)
            if co_base:
                post['input'] += g * (post['V'] - E)
            else:
                post['input'] += g

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='alpha_synapse',
                      requires=requires,
                      ST=ST,
                      steps=(update, output),
                      mode=mode)
