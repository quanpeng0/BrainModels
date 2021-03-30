# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
#import bpmodels
import pdb
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
    ============= ============== ======== =======================================

    **Synapse Variables**    

    An object of synapse class record those variables for each synapse:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    s                  0.                Gating variable.

    g                  0.                Synapse conductance on post-synaptic neuron.
    ================== ================= =========================================================

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
        self.s = bp.backend.zeros(self.size)
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
            #pdb.set_trace()
            self.g.push(i,self.g_max * self.s[i])
            post_id = self.post_ids[i]
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)


class GABAa1_mat(bp.TwoEndConn):
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
    ============= ============== ======== =======================================

    **Synapse Variables**    

    An object of synapse class record those variables for each synapse:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    s                  0.                Gating variable.

    g                  0.                Synapse conductance on post-synaptic neuron.
    ================== ================= =========================================================

    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """
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
        self.size = bp.backend.shape(self.conn_mat)

        # data
        self.s = bp.backend.zeros(self.size)
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
        self.post.input -= bp.backend.sum(g, axis=0) * (self.post.V - self.E)


class GABAa2_vec(bp.TwoEndConn):
    target_backend = 'general'

    def __init__(self, pre, post, conn, delay = 0.,
                 g_max=0.04, E=-80., alpha=0.53, 
                 beta=0.18, T=1., T_duration=1.,
                 **kwargs):
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.T_duration = T_duration

        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = self.conn.requires(
            'pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size = self.size, delay_time = delay)
        self.t_last_pre_spike = bp.backend.ones(self.size) * -1e7
        super(GABAa2_vec, self).__init__(pre = pre, post = post, **kwargs)

    @staticmethod
    @bp.odeint
    def integral(s, t, TT, alpha, beta):
        dsdt = alpha * TT * (1 - s) - beta * s
        return dsdt

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            post_id = self.post_ids[i]
            if self.pre.spike[pre_id]:
                self.t_last_pre_spike[i] = _t
            T = ((_t - self.t_last_pre_spike[i]) < self.T_duration) * self.T
            self.s[i] = self.integral(self.s[i], _t, T, self.alpha, self.beta)
            self.g.push(i, self.s[i] * self.g_max)
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)
                

class GABAa2_mat(bp.TwoEndConn):
    target_backend = 'general'

    def __init__(self, pre, post, conn, delay = 0.,
                 g_max=0.04, E=-80., alpha=0.53, 
                 beta=0.18, T=1., T_duration=1.,
                 **kwargs):
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.T_duration = T_duration

        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size = self.size, delay_time = delay)
        self.t_last_pre_spike = bp.backend.ones(self.size) * -1e7
        super(GABAa2_mat, self).__init__(pre = pre, post = post, **kwargs)

    @staticmethod
    @bp.odeint
    def integral(s, t, TT, alpha, beta):
        dsdt = alpha * TT * (1 - s) - beta * s
        return dsdt

    def update(self, _t):
        spike = bp.backend.reshape(self.pre.spike, (-1, 1)) * self.conn_mat
        self.t_last_pre_spike = bp.backend.where(spike, _t, self.t_last_pre_spike)
        TT = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
        self.s = self.integral(self.s, _t, TT, self.alpha, self.beta)
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.backend.sum(self.g.pull(), 0) * (self.post.V - self.E)

import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt

duration = 100.
dt = 0.02
bp.backend.set('numpy', dt=dt)
size = 10
neu_pre = brainmodels.neurons.LIF(size, monitors = ['V', 'input', 'spike'])
neu_pre.V_rest = -65.
neu_pre.V_th = -50.
neu_pre.V_reset = -70.
neu_pre.V = bp.backend.ones(size) * -65.
neu_pre.t_refractory = 0.
neu_post = brainmodels.neurons.LIF(size, monitors = ['V', 'input', 'spike'])
neu_post.V_rest = -65.
neu_post.V_th = -50.
neu_post.V_reset = -70.
neu_post.V = bp.backend.ones(size) * -65.
neu_post.t_refractory = 0.

syn_GABAa = GABAa2_vec(pre = neu_pre, post = neu_post, 
                       conn = bp.connect.All2All(),
                       delay = 10., monitors = ['s'])

net = bp.Network(neu_pre, syn_GABAa, neu_post)
net.run(duration, inputs = (neu_pre, 'input', 21.), report = True)

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
plt.plot(ts, neu_pre.mon.V[:, 0], label='pre.V')
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
