# -*- coding: utf-8 -*-
import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt
import numpy as np


class fr_neu(bp.NeuGroup):
    target_backend = 'general'

    def __init__(self, size, **kwargs):
        self.r = bp.ops.zeros(size)
        super(fr_neu, self).__init__(size=size, **kwargs)

    def update(self, _t):
        self.r = self.r


if __name__ == "__main__":
    # set params
    neu_pre_num = 2
    neu_post_num = 2
    dt = 0.02
    bp.backend.set('numpy', dt=dt)

    # build network
    neu_pre = fr_neu(neu_pre_num, monitors=['r'], show_code=True)
    neu_post = fr_neu(neu_post_num, monitors=['r'], show_code=True)

    syn = brainmodels.numba_backend.synapses.Oja(
        pre=neu_pre, post=neu_post,
        conn=bp.connect.All2All(), monitors=['w'], show_code=True
    )

    net = bp.Network(neu_pre, syn, neu_post)

    # create input
    current_mat_in = []
    current_mat_out = []
    current1, _ = bp.inputs.constant_current(
        [(2., 20.), (0., 20.)] * 3 + [(0., 20.), (0., 20.)] * 2)
    current2, _ = bp.inputs.constant_current([(2., 20.), (0., 20.)] * 5)
    current3, _ = bp.inputs.constant_current([(2., 20.), (0., 20.)] * 5)
    current_mat_in = np.vstack((current1, current2))
    current_mat_out = current3
    current_mat_out = np.vstack((current_mat_out, current3))

    # simulate network
    net.run(duration=200.,
            inputs=[(neu_pre, 'r', current_mat_in.T, '='),
                    (neu_post, 'r', current_mat_out.T)
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
