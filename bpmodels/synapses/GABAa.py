# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
import sys


def get_GABAa1(g_max=0.4, E=-80., tau_decay=6., mode='vector'):
    """
    GABAa conductance-based synapse model.

    .. math::

        I_{syn}&= - \\bar{g}_{max} s (V - E)

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k}\\delta(t-t-{j}^{k})


    **Synapse Parameters**

    ============= ============== ======== =======================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ---------------------------------------
    g_max         0.4            \        Maximum synapse conductance.

    E             -80.           \        Reversal potential of synapse.

    tau_decay     6.             ms       Time constant of gating variable decay.

    mode          'vector'       \        Data structure of ST members.
    ============= ============== ======== =======================================

    Returns:
        bp.SynType: return description of GABAa synapse model.


    **Synapse State**

    ST refers to synapse state, members of ST are listed below:

    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    s               0.                Gating variable.

    g               0.                Synapse conductance on post-synaptic neuron.
    =============== ================= =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """

    ST = bp.types.SynState('s', 'g')

    requires = dict(
        pre=bp.types.NeuState('spike'),
        post=bp.types.NeuState('V', 'input'),
    )

    @bp.integrate
    def int_s(s, t):
        return - s / tau_decay

    if mode == 'scalar':

        def update(ST, _t, pre):
            s = int_s(ST['s'], _t)
            s += pre['spike']
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, _t, post):
            I_syn = ST['g'] * (post['V'] - E)
            post['input'] -= I_syn

    elif mode == 'vector':

        requires['pre2syn'] = bp.types.ListConn()
        requires['post_slice_syn'] = bp.types.Array(dim=2)

        def update(ST, pre, pre2syn):
            s = int_s(ST['s'], 0.)
            for pre_id in np.where(pre['spike'] > 0.)[0]:
                syn_ids = pre2syn[pre_id]
                s[syn_ids] += 1
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post, post_slice_syn):
            post_cond = np.zeros(len(post_slice_syn), dtype=np.float_)
            for i, [s, e] in enumerate(post_slice_syn):
                post_cond[i] = np.sum(ST['g'][s:e])
            post['input'] -= post_cond * (post['V'] - E)

    elif mode == 'matrix':

        requires['conn_mat'] = bp.types.MatConn(
            help='Connectivity matrix with shape of (num_pre, num_post)')

        def update(ST, _t, pre, conn_mat):
            s = int_s(ST['s'], _t)
            for i in range(pre['spike'].shape[0]):
            	if pre['spike'][i] > 0.:
	                s[i] += conn_mat[i]
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post):
            g = np.sum(ST['g'], axis=0)
            post['input'] -= g * (post['V'] - E)
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='GABAa_synapse',
                      ST=ST,
                      requires=requires,
                      steps=(update, output),
                      mode=mode)


def get_GABAa2(g_max=0.04, E=-80., alpha=0.53, beta=0.18, T=1., T_duration=1., mode='vector'):
    """
    GABAa conductance-based synapse model (markov form).

    .. math::

        I_{syn}&= - \\bar{g}_{max} s (V - E)

        \\frac{d s}{d t}&=\\alpha[T](1-s) - \\beta s

    ST refers to synapse state, members of ST are listed below:

    ================ ================= =========================================================
    **Member name**  **Initial Value** **Explanation**
    ---------------- ----------------- ---------------------------------------------------------
    s                0.                Gating variable.

    g                0.                Synapse conductance on post-synaptic neuron.

    t_last_pre_spike -1e7              Last spike time stamp of pre-synaptic neuron.
    ================ ================= =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    References:
        .. [1] Destexhe, Alain, and Denis Paré. "Impact of network activity
               on the integrative properties of neocortical pyramidal neurons
               in vivo." Journal of neurophysiology 81.4 (1999): 1531-1547.
    """

    ST = bp.types.SynState('s', 'g', t_last_pre_spike=-1e7)

    requires = dict(
        pre=bp.types.NeuState(
            ['spike'], help="Pre-synaptic neuron state must have 'spike' item"),
        post=bp.types.NeuState(
            ['V', 'input'], help="Post-synaptic neuron state must have 'V' and 'input' item"),
    )

    @bp.integrate
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    if mode == 'scalar':

        def update(ST, _t, pre, post):
            if pre['spike'] > 0.:
                ST['t_last_pre_spike'] = _t
            TT = ((_t - ST['t_last_pre_spike']) < T_duration) * T
            s = int_s(ST['s'], _t, TT)
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post):
            post['input'] -= ST['g'] * (post['V'] - E)

    elif mode == 'vector':

        requires['pre2syn'] = bp.types.ListConn(
            help="Pre-synaptic neuron index -> synapse index")
        requires['post_slice_syn'] = bp.types.Array(dim=2)

        def update(ST, _t, pre, pre2syn):
            for pre_id in np.where(pre['spike'] > 0.)[0]:
                syn_ids = pre2syn[pre_id]
                ST['t_last_pre_spike'][syn_ids] = _t
            TT = ((_t - ST['t_last_pre_spike']) < T_duration) * T
            s = int_s(ST['s'], _t, TT)
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post, post_slice_syn):
            post_cond = np.zeros(len(post_slice_syn), dtype=np.float_)
            for i, [s, e] in enumerate(post_slice_syn):
                post_cond[i] = np.sum(ST['g'][s:e])
            post['input'] -= post_cond * (post['V'] - E)

    elif mode == 'matrix':
        def update(ST, _t, pre):
            spike_idxs = np.where(pre['spike'] > 0.)[0]
            ST['t_last_pre_spike'][spike_idxs] = _t
            TT = ((_t - ST['t_last_pre_spike']) < T_duration) * T
            s = int_s(ST['s'], _t, TT)
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post):
            g = np.sum(ST['g'], axis=0)
            post['input'] -= g * (post['V'] - E)
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='GABAa_synapse',
                      ST=ST,
                      requires=requires,
                      steps=(update, output),
                      mode=mode)
