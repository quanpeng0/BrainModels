import brainmodels as bm
import brainpy as bp
import matplotlib.pyplot as plt

bp.backend.set(backend='numpy', dt=0.1)

pre = bm.neurons.LIF(1, monitors=['spike'])
post = bm.neurons.LIF(1, monitors=['spike'])

# pre before post
duration = 60.
(I_pre, _) = bp.inputs.constant_current([(0, 5), (30, 15), 
                                         (0, 5), (30, 15), 
                                         (0, duration-40)])
(I_post, _) = bp.inputs.constant_current([(0, 7), (30, 15), 
                                          (0, 5), (30, 15), 
                                          (0, duration-7-35)])

syn = bm.synapses.STDP(pre=pre, post=post, conn=bp.connect.All2All(), monitors=['s', 'A_s', 'A_t', 'w'])
net = bp.Network(pre, syn, post)
net.run(duration, inputs=[(pre, 'input', I_pre), (post, 'input', I_post)], report=True)

# plot
fig, gs = bp.visualize.get_figure(3, 1)

fig.add_subplot(gs[0, 0])
plt.plot(net.ts, syn.mon.w[:, 0], label='w')
plt.legend()

fig.add_subplot(gs[2, 0])
plt.plot(net.ts, 2*pre.mon.spike[:, 0], label='pre_spike')
plt.plot(net.ts, 2*post.mon.spike[:, 0], label='post_spike')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(net.ts, syn.mon.s[:, 0], label='s')
plt.legend()

plt.xlabel('Time (ms)')
plt.show()