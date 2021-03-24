# -*- coding: utf-8 -*-
#import bpmodels
import matplotlib.pyplot as plt
import brainpy as bp
import numpy as np
import sys
import pdb
from numba import prange


class Oja(bp.TwoEndConn):

    target_backend = ['numpy', 'numba']

    def __init__(self, pre, post, conn, delay=0., 
                 gamma=0.005, w_max=1., w_min=0.,
                 **kwargs):
        # params
        self.gamma = gamma
        self.w_max = w_max
        self.w_min = w_min
        # no delay in firing rate models

        # conns
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        #data
        self.w = bp.backend.ones(self.size) * 0.05

        super(Oja, self).__init__(pre = pre, post = post, **kwargs)
    
    @staticmethod
    @bp.odeint()
    def integral(w, t, gamma, r_pre, r_post):
        return gamma * (r_post * r_pre - r_post * r_post * w)

    def update(self, _t):
        post_r = bp.backend.zeros(self.post.size[0])
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            post_id = self.post_ids[i]
            #pdb.set_trace()
            add = self.w[i] * self.pre.r[pre_id]
            add = np.sum(add)
            post_r[post_id] += add
            self.w[i] = self.integral(self.w[i], _t, self.gamma,
                                      self.pre.r[pre_id], self.post.r[post_id])
        self.post.r = post_r


class fr_neu(bp.NeuGroup):
    target_backend = 'general'

    def __init__(self, size, **kwargs):
        self.r = bp.backend.zeros(size)
        super(fr_neu, self).__init__(size = size, **kwargs)
    
    def update(self, _t):
        self.r = self.r


if __name__ == "__main__":
    # set params
    neu_pre_num = 2
    neu_post_num = 3
    dt = 0.02
    bp.backend.set('numpy', dt = dt)

    # build network
    neu_pre = fr_neu(neu_pre_num, monitors=['r'], show_code=True)
    neu_post = fr_neu(neu_post_num, monitors=['r'], show_code=True)

    syn = Oja(pre = neu_pre, post = neu_post, 
              conn = bp.connect.All2All(), monitors = ['w'], show_code=True)

    net = bp.Network(neu_pre, syn, neu_post)

    # create input
    current_mat_in = []
    current_mat_out = []
    current1, _ = bp.inputs.constant_current([(0., 20.), (2., 20.)] * 5)
    #current1, _ = bp.inputs.constant_current(
    #    [(2., 20.), (0., 20.)] * 3 + [(0., 20.), (0., 20.)] * 2)
    current2, _ = bp.inputs.constant_current([(2., 20.), (0., 20.)] * 5)
    current3, _ = bp.inputs.constant_current([(2., 20.), (0., 20.)] * 5)
    current_mat_in = np.vstack((current1, current2))
    #current_mat_out = np.vstack((current3, current3))
    current_mat_out = current3
    current_mat_out = np.vstack((current_mat_out, current3))
    current_mat_out = np.vstack((current_mat_out, current3))

    # simulate network
    net.run(duration=200., 
            inputs=[(neu_pre, 'r', current_mat_in.T, '='),
                    (neu_post, 'r', 2.)
                    ],
            report=True)

    # paint
    fig, gs = bp.visualize.get_figure(4, 1, 3, 12)

    fig.add_subplot(gs[0, 0])
    plt.plot(net.ts, neu_pre.mon.r[:, 0], label='pre r1')
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(net.ts, neu_pre.mon.r[:, 1], label='pre r2')
    plt.legend()

    fig.add_subplot(gs[2, 0])
    plt.plot(net.ts, neu_post.mon.r[:, 0], label='post r')
    plt.ylim([0, 4])
    plt.legend()

    fig.add_subplot(gs[3, 0])
    plt.plot(net.ts, syn.mon.w[:, 0], label='syn.w1')
    plt.plot(net.ts, syn.mon.w[:, 3], label='syn.w2')
    plt.legend()
    plt.show()
    
'''
    elif mode == 'matrix':
        def update(ST, _t, pre, post, conn_mat):
            post['r'] = np.dot(pre['r'], conn_mat * ST['w'])
            expand_pre = np.expand_dims(pre['r'], axis=1) \
                .repeat(post['r'].shape[0], axis=1)
            expand_post = np.expand_dims(post['r'], axis=1) \
                .reshape(1, -1) \
                .repeat(pre['r'].shape[0], axis=0)
            ST['w'] = int_w(ST['w'], _t, expand_pre, expand_post)

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='Oja_synapse',
                      ST=ST,
                      requires=requires,
                      steps=update,
                      mode=mode)
'''
'''

def get_Oja(gamma=0.005, w_max=1., w_min=0., mode='vector'):
    """
    Oja's learning rule.

    .. math::

        \\frac{d w_{ij}}{dt} = \\gamma(\\upsilon_i \\upsilon_j - w_{ij}\\upsilon_i ^ 2)


    **Learning Rule Parameters**

    ============= ============== ======== ================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------
    gamma         0.005          \        Learning rate.

    w_max         1.             \        Maximal possible synapse weight.

    w_min         0.             \        Minimal possible synapse weight.

    mode          'vector'       \        Data structure of ST members.
    ============= ============== ======== ================================

    Returns:
        bp.Syntype: return description of synapse with Oja's rule.


    **Learning Rule State**

    ST refers to synapse state (note that Oja learning rule can be implemented as synapses),
    members of ST are listed below:

    ================ ================= =========================================================
    **Member name**  **Initial Value** **Explanation**
    ---------------- ----------------- ---------------------------------------------------------
    w                0.05              Synapse weight.

    output_save      0.                Temporary save synapse output value until post-synaptic
                                       neuron get the value after delay time.
    ================ ================= =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).        

    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """

    ST = bp.types.SynState('output_save', w=0.05)

    requires = dict(
        pre=bp.types.NeuState(['r']),
        post=bp.types.NeuState(['r']),
    )

    @bp.integrate
    def int_w(w, t, r_pre, r_post):
        return gamma * (r_post * r_pre - r_post * r_post * w)

    if mode == 'scalar':
        raise ValueError("mode of function '%s' can not be '%s'." %
                         (sys._getframe().f_code.co_name, mode))

    elif mode == 'vector':

        requires['post2syn'] = bp.types.ListConn()
        requires['post2pre'] = bp.types.ListConn()

        def update(ST, _t, pre, post, post2pre, post2syn):
            for i in range(len(post2pre)):
                pre_ids = post2pre[i]
                syn_ids = post2syn[i]
                post['r'] = np.sum(ST['w'][syn_ids] * pre['r'][pre_ids])
                ST['w'][syn_ids] = int_w(
                    ST['w'][syn_ids], _t,  pre['r'][pre_ids], post['r'][i])

    elif mode == 'matrix':

        requires['conn_mat'] = bp.types.MatConn(
            help='Connectivity matrix with shape of (num_pre, num_post)')

        def update(ST, _t, pre, post, conn_mat):
            post['r'] = np.dot(pre['r'], conn_mat * ST['w'])
            expand_pre = np.expand_dims(pre['r'], axis=1) \
                .repeat(post['r'].shape[0], axis=1)
            expand_post = np.expand_dims(post['r'], axis=1) \
                .reshape(1, -1) \
                .repeat(pre['r'].shape[0], axis=0)
            ST['w'] = int_w(ST['w'], _t, expand_pre, expand_post)

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='Oja_synapse',
                      ST=ST,
                      requires=requires,
                      steps=update,
                      mode=mode)
'''