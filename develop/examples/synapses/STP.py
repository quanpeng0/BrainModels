import brainpy as bp
import bpmodels
import matplotlib.pyplot as plt

hh1 = bpmodels.neurons.HH(1, monitors=['V'])
hh2 = bpmodels.neurons.HH(1, monitors=['V'])

# f
syn= bpmodels.synapses.STP(U=0.1, tau_d=100, tau_f=2000., pre=hh1, post=hh2, conn=bp.connect.All2All(),
                    delay=0., monitors=['s', 'u', 'x'] )
net = bp.Network(hh1, hh2, syn)
net.run(duration=300., inputs=(hh1, 'input', 10.))

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


# d
syn= bpmodels.STP(U=0.55, tau_d=1500, tau_f=50., pre=hh1, post=hh2, conn=bp.connect.All2All(),
                    delay=0., monitors=['s', 'u', 'x'] )
net = bp.Network(hh1, hh2, syn)
net.run(duration=300., inputs=(hh1, 'input', 10.))

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
