# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np


def get_exponential(tau_decay=8., g_max=.1, E=0., mode='scalar', co_base = False):
    '''
    Exponential decay synapse model.

    .. math::

         \\frac{d s}{d t} = -\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})

    For conductance-based (co-base=True):

    .. math::
    
        I_{syn}(t) = g_{syn} (t) (V(t)-E_{syn})


    For current-based (co-base=False):

    .. math::
    
        I(t) = \\bar{g} s (t)

    ST refers to synapse state, members of ST are listed below:
    
    ================ ================== =========================================================
    **Member name**  **Initial values** **Explanation**
    ---------------- ------------------ ---------------------------------------------------------    
    s                  0                  Gating variable.

    g                  0                  Synapse conductance on the post-synaptic neuron.
                             
    ================ ================== =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        tau_decay (float): The time constant of decay.
        g_max (float): Synaptic weight.
        mode (string): data structure of ST members.

    Returns:
        bp.Neutype: return description of exponential synapse model.

    References:
        .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw. 
                "The Synapse." Principles of Computational Modelling in Neuroscience. 
                Cambridge: Cambridge UP, 2011. 172-95. Print.
    '''

    @bp.integrate
    def int_s(s, t):
        return - s / tau_decay

    ST = bp.types.SynState(('s', 'g'), help='synapse state.')

    requires = {
        'pre': bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "spike" item.'),
        'post': bp.types.NeuState(['V', 'input'], help='Post-synaptic neuron state must have "V" and "input" item.')
    }

    if mode == 'scalar':
        def update(ST, _t, pre):
            s = int_s(ST['s'], _t)
            s += pre['spike']
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
            s = int_s(ST['s'], _t)
            spike_idx = np.where(pre['spike'] > 0.)[0]
            for i in spike_idx:
                syn_idx = pre2syn[i]
                s[syn_idx] += 1.
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
            s = int_s(ST['s'], _t)
            s += pre['spike'].reshape((-1, 1)) * conn_mat
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


    return bp.SynType(name='exponential_synapse',
                      ST=ST, requires=requires,
                      steps=(update, output),
                      mode = mode)
