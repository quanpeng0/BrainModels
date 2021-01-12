# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
import sys


def get_STDP(g_max=0.10, E=0., tau_decay=10., tau_s=10., tau_t=10.,
             w_min=0., w_max=20., delta_A_s=0.5, delta_A_t=0.5, mode='vector'):
    """
    Spike-time dependent plasticity.

    .. math::

        \\frac{d A_{source}}{d t}&=-\\frac{A_{source}}{\\tau_{source}}

        \\frac{d A_{target}}{d t}&=-\\frac{A_{target}}{\\tau_{target}}

    After a pre-synaptic spike:

    .. math::      

        g_{post}&= g_{post}+w

        A_{source}&= A_{source} + \\Delta A_{source}

        w&= min([w-A_{target}]^+, w_{max})

    After a post-synaptic spike:

    .. math::

        A_{target}&= A_{target} + \\Delta A_{target}

        w&= min([w+A_{source}]^+, w_{max})

    **Learning Rule Parameters**

    ============= ============== ======== =================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -----------------------------------------------------------------
    g_max         0.1            \        Maximum conductance.

    E             0.             \        Reversal potential.

    tau_decay     10.            ms       Time constant of decay.

    tau_s         10.            ms       Time constant of source neuron 

                                          (i.e. pre-synaptic neuron)

    tau_t         10.            ms       Time constant of target neuron 

                                          (i.e. post-synaptic neuron)

    w_min         0.             \        Minimal possible synapse weight.

    w_max         20.            \        Maximal possible synapse weight.

    delta_A_s     0.5            \        Change on source neuron traces elicited by 

                                          a source neuron spike.

    delta_A_t     0.5            \        Change on target neuron traces elicited by 

                                          a target neuron spike.

    mode          'vector'       \        Data structure of ST members.
    ============= ============== ======== =================================================================

    Returns:
        bp.Syntype: return description of STDP.


    **Learning Rule State**

    ST refers to synapse state (note that STDP learning rule can be implemented as synapses),
    members of ST are listed below:

    ================ ================= =========================================================
    **Member name**  **Initial Value** **Explanation**
    ---------------- ----------------- ---------------------------------------------------------
    A_s              0.                Source neuron trace.

    A_t              0.                Target neuron trace.

    g                0.                Synapse conductance on post-synaptic neuron.

    w                0.                Synapse weight.
    ================ ================= =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    References:
        .. [1] Stimberg, Marcel, et al. "Equation-oriented specification of neural models for
               simulations." Frontiers in neuroinformatics 8 (2014): 6.
    """

    ST = bp.types.SynState('A_s', 'A_t', 'g', 'w')

    requires = dict(
        pre=bp.types.NeuState(['spike'], help='Pre-synaptic neuron state \
                                               must have "spike" item.'),
        post=bp.types.NeuState(['V', 'input', 'spike'],
                               help='Pre-synaptic neuron state must \
                                     have "V", "input" and "spike" item.'),
    )

    @bp.integrate
    def int_A_s(A_s, t):
        return -A_s / tau_s

    @bp.integrate
    def int_A_t(A_t, t):
        return -A_t / tau_t

    @bp.integrate
    def int_g(g, t):
        return -g / tau_decay

    if mode == 'scalar':
        def my_relu(w):
            return w if w > 0 else 0

        def update(ST, _t, pre, post):
            A_s = int_A_s(ST['A_s'], _t)
            A_t = int_A_t(ST['A_t'], _t)
            g = int_g(ST['g'], _t)
            w = ST['w']
            if pre['spike']:
                g += ST['w']
                A_s = A_s + delta_A_s
                w = np.clip(my_relu(ST['w'] - A_t), w_min, w_max)
            if post['spike']:
                A_t = A_t + delta_A_t
                w = np.clip(my_relu(ST['w'] + A_s), w_min, w_max)
            ST['A_s'] = A_s
            ST['A_t'] = A_t
            ST['g'] = g
            ST['w'] = w

        @bp.delayed
        def output(ST, post):
            I_syn = - g_max * ST['g'] * (post['V'] - E)
            post['input'] += I_syn

    elif mode == 'vector':

        requires['pre2syn'] = bp.types.ListConn(
            help='Pre-synaptic neuron index -> synapse index')
        requires['post2syn'] = bp.types.ListConn(
            help='Post-synaptic neuron index -> synapse index')

        def update(ST, _t, pre, post, pre2syn, post2syn):
            A_s = int_A_s(ST['A_s'], _t)
            A_t = int_A_t(ST['A_t'], _t)
            g = int_g(ST['g'], _t)
            w = ST['w']
            for i in np.where(pre['spike'] > 0.)[0]:
                syn_ids = pre2syn[i]
                g[syn_ids] += ST['w'][syn_ids]
                A_s[syn_ids] = A_s[syn_ids] + delta_A_s
                w[syn_ids] = np.clip(ST['w'][syn_ids] -
                                     ST['A_t'][syn_ids], w_min, w_max)
            for i in np.where(post['spike'] > 0.)[0]:
                syn_ids = post2syn[i]
                A_t[syn_ids] = A_t[syn_ids] + delta_A_t
                w[syn_ids] = np.clip(ST['w'][syn_ids] +
                                     ST['A_s'][syn_ids], w_min, w_max)
            ST['A_s'] = A_s
            ST['A_t'] = A_t
            ST['g'] = g
            ST['w'] = w

        @bp.delayed
        def output(ST, post, post_slice_syn):
            post_cond = np.zeros(len(post_slice_syn), dtype=np.float_)
            for i, [s, e] in enumerate(post_slice_syn):
                post_cond[i] = np.sum(g_max * ST['g'][s:e])
            post['input'] -= post_cond * (post['V'] - E)

    elif mode == 'matrix':
        requires['conn_mat'] = bp.types.MatConn(
            help='Connectivity matrix with shape of (num_pre, num_post)')

        def update(ST, _t, pre, post, conn_mat):
            A_s = int_A_s(ST['A_s'], _t)
            A_t = int_A_t(ST['A_t'], _t)
            g = int_g(ST['g'], _t)
            w = ST['w']

            pre_spike = pre['spike'].reshape((-1, 1)) * conn_mat
            g += pre_spike * ST['w']
            A_s += pre_spike * delta_A_s
            w -= pre_spike * ST['A_t']
            w = np.clip(w, w_min, w_max)

            post_spike = (post['spike'].reshape((-1, 1)) * conn_mat.T).T
            A_t += post_spike * delta_A_t
            w += post_spike * ST['A_s']
            w = np.clip(w, w_min, w_max)

            ST['A_s'] = A_s
            ST['A_t'] = A_t
            ST['g'] = g
            ST['w'] = w

        @bp.delayed
        def output(ST, post):
            g = g_max * np.sum(ST['g'], axis=0)
            post['input'] -= g * (post['V'] - E)

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='STDP_synapse',
                      ST=ST,
                      requires=requires,
                      steps=(update, output),
                      mode=mode)
