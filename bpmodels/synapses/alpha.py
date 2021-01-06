# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np


def get_alpha(tau_decay = 2., g_max=.2, E=0., mode='scalar', co_base = False):

    """
    Alpha synapse.

    .. math::

        &\\frac {ds} {dt} = x
        
        &\\tau_{decay}^2 \\frac {dx} {dt} = - 2  \\tau_{decay}  \\frac {x}
        - s + \\sum \\delta(t-t^f)


    For conductance-based (co-base=True):

    .. math::
    
        I_{syn}(t) = g_{syn} (t) (V(t)-E_{syn})


    For current-based (co-base=False):

    .. math::
    
        I(t) = \\bar{g} s (t)


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

    Args:
        tau_decay (float): The time constant of decay.
        g_max (float): The peak conductance change in µmho (µS).
        E (float): The reversal potential for the synaptic current. (only for conductance-based)

    Returns:
        bp.Neutype

    References:
        .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw. 
                "The Synapse." Principles of Computational Modelling in Neuroscience. 
                Cambridge: Cambridge UP, 2011. 172-95. Print.
    """

    ST=bp.types.SynState(('g','s','x'), help='The conductance defined by exponential function.')

    requires = {
        'pre': bp.types.NeuState(['spike'], help='pre-synaptic neuron state must have "V"'),
        'post': bp.types.NeuState(['input', 'V'], help='post-synaptic neuron state must include "input" and "V"')
    }

    @bp.integrate
    def int_s(s, t, x):
        return x

    @bp.integrate
    def int_x(x, t, s):
        return (-2 * tau_decay * x - s ) / (tau_decay**2)

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
                post['input'] += ST['g']* (post['V'] - E)
            else:
                post['input'] += ST['g']

    elif mode == 'vector':
        requires['pre2syn']=bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index')
        requires['post2syn']=bp.types.ListConn(help='Post-synaptic neuron index -> synapse index')

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
        def output(ST, post, post2syn):
            for post_id, syn_id in enumerate(post2syn):
                if co_base:
                    post['input'][post_id] += np.sum(ST['g'][syn_id])* (post['V'] - E)
                else:
                    post['input'][post_id] += np.sum(ST['g'][syn_id])

    elif mode == 'matrix':
        requires['conn_mat']=bp.types.MatConn()

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
                post['input'] += g* (post['V'] - E)
            else:
                post['input'] += g

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='alpha_synapse',
                 requires=requires,
                 ST=ST,
                 steps=(update, output),
                 mode = mode)