import brainpy as bp
import brainmodels as bm
import matplotlib.pyplot as plt

backend = 'numpy'
bp.backend.set(backend=backend, dt=.01)
bm.set_backend(backend=backend)

# STD
# U = 0.55
# tau_d = 100.
# tau_f = 2.

# STF
U = 0.1
tau_d = 10.
tau_f = 100.

# run
neu1 = bm.neurons.LIF(1, monitors=['V'])
neu2 = bm.neurons.LIF(1, monitors=['V'])

syn = bm.synapses.STP(U=U, tau_d=tau_d, tau_f=tau_f, pre=neu1, post=neu2,
                      conn=bp.connect.All2All(), monitors=['s', 'u', 'x'])
net = bp.Network(neu1, syn, neu2)
net.run(100., inputs=(neu1, 'input', 28.))

# plot
fig, gs = bp.visualize.get_figure(2, 1, 3, 7)

fig.add_subplot(gs[0, 0])
plt.plot(net.ts, syn.mon.u[:, 0], label='u')
plt.plot(net.ts, syn.mon.x[:, 0], label='x')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(net.ts, syn.mon.s[:, 0], label='s')
plt.legend()

plt.xlabel('Time (ms)')
plt.show()
