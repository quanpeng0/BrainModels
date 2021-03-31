
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

syn_GABAa = brainmodels.synapses.GABAa2(pre = neu_pre, post = neu_post, 
                       conn = bp.connect.All2All(),
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
plt.plot(ts, neu_pre.mon.V[:, 0], label='pre.V')
plt.legend()

fig.add_subplot(gs[1, 1])
plt.plot(ts, neu_pre.mon.spike[:, 0], label='pre.spike')
plt.legend()

plt.show()
