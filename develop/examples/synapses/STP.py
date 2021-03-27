import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt


LIF = brainmodels.neurons.get_LIF(V_rest=-65., V_reset=-65., V_th=-55)

neu = bp.NeuGroup(LIF, 1, monitors=['V'])
neu.ST['V'] = -65.

duration=300
I = bp.inputs.spike_current([10, 90, 150, 200, 220], bp.profile._dt, 1., duration=duration)

# f
syn_model = brainmodels.synapses.get_STP(U=0.1, tau_d=100, tau_f=2000.)
syn = bp.SynConn(syn_model, pre_group=neu, post_group=neu, 
                conn=bp.connect.All2All(), monitors=['u', 'x', 's'])
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(syn, 'pre.spike', I, '='))

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
syn_model = brainmodels.synapses.get_STP(U=0.55, tau_d=1500., tau_f=50.)
syn = bp.SynConn(syn_model, pre_group=neu, post_group=neu, 
                conn=bp.connect.All2All(), monitors=['u', 'x', 's'])
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(syn, 'pre.spike', I, '='))

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