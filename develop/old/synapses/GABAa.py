# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
import bpmodels
from numba import prange
import matplotlib.pyplot as plt

class GABAa1_vec(bp.TwoEndConn):    
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
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numba-cuda']

    def __init__(self, pre, post, conn, delay=0., 
                 g_max=0.4, E=-80., tau_decay=6., 
                 **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.tau_decay = tau_decay
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # data
        self.s = bp.ops.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(GABAa1_vec, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def integral(s, t, tau_decay):
        return - s / tau_decay

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            self.s[i] = self.integral(self.s[i], _t, self.tau_decay)
            self.s[i] += self.pre.spike[pre_id]
            self.g.push(i,self.g_max * self.s[i])
            post_id = self.post_ids[i]
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)


class GABAa1_mat(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel']

    def __init__(self, pre, post, conn, delay=0., 
                 g_max=0.4, E=-80., tau_decay=6., 
                 **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.tau_decay = tau_decay
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        # data
        self.s = bp.ops.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(GABAa1_mat, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint
    def int_s(s, t, tau_decay):
        return - s / tau_decay

    def update(self, _t):
        self.s = self.int_s(self.s, _t, self.tau_decay)
        for i in range(self.pre.size[0]):
            if self.pre.spike[i] > 0:
                self.s[i] += self.conn_mat[i]
        self.g.push(self.g_max * self.s)
        g=self.g.pull()
        self.post.input -= bp.ops.sum(g, axis=0) * (self.post.V - self.E)


class LIF(bp.NeuGroup):
    target_backend = 'general'  #TODO: the relation between backend and relization.

    def __init__(self, size, V_rest = 0., V_reset= -5., 
                 V_th = 20., R = 1., tau = 10., 
                 t_refractory = 5., **kwargs):
        
        # parameters
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.t_refractory = t_refractory

        # variables
        self.V = bp.ops.zeros(size)
        self.input = bp.ops.zeros(size)
        self.spike = bp.ops.zeros(size)
        self.refractory = bp.ops.zeros(size)
        self.t_last_spike = bp.ops.ones(size) * -1e7

        super(LIF, self).__init__(size = size, **kwargs)

    @staticmethod
    @bp.odeint()
    def integral(V, t, I_ext, V_rest, R, tau):  # integrate u(t)
        return (- (V - V_rest) + R * I_ext) / tau
    
    def update(self, _t):
        # update variables
        #pdb.set_trace()
        #TODO: may I consider which backend and judge if I need prange?
        for i in prange(self.size[0]):
            refractory = (_t - self.t_last_spike[i] <= self.t_refractory)
            if refractory:
                spike = 0.
            else:
                V = self.integral(self.V[i], _t, self.input[i], self.V_rest, self.R, self.tau)
                spike = (V >= self.V_th)
                if spike:
                    V = self.V_rest
                    self.t_last_spike[i] = _t
                self.V[i] = V
            self.spike[i] = spike
            self.refractory[i] = refractory
            self.input[i] = 0.  # reset input here or it will be brought to next step


if __name__ == "__main__":

    duration = 100.
    dt = 0.02
    bp.backend.set('numpy', dt=dt)
    size = 10
    neu_pre = LIF(size, monitors = ['V', 'input', 'spike'])
    neu_pre.V_rest = -65.
    neu_pre.V_th = -50.
    neu_pre.V = bp.ops.ones(size) * -65.
    neu_pre.t_refractory = 0.
    neu_post = LIF(size, monitors = ['V', 'input', 'spike'])
    neu_post.V_rest = -65.
    neu_post.V = bp.ops.ones(size) * -65.

    syn_GABAa = GABAa1_mat(pre = neu_pre, post = neu_post, 
                           conn = bp.connect.One2One(),
                           delay = 10., monitors = ['s'])

    net = bp.Network(neu_pre, syn_GABAa, neu_post)
    net.run(duration, inputs = (neu_pre, 'input', 16.), report = True)

    # paint gabaa
    ts = net.ts
    fig, gs = bp.visualize.get_figure(2, 2, 5, 6)

    #print(gabaa.mon.s.shape)
    fig.add_subplot(gs[0, 0])
    plt.plot(ts, syn_GABAa.mon.s[:, 0], label='s')
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(ts, neu_post.mon.V[:, 0], label='post.V')
    plt.legend()

    fig.add_subplot(gs[0, 1])
    plt.plot(ts, neu_post.mon.input[:, 0], label='post.input')
    plt.legend()

    fig.add_subplot(gs[1, 1])
    plt.plot(ts, neu_pre.mon.spike[:, 0], label='pre.spike')
    plt.legend()

    plt.show()

'''

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

'''