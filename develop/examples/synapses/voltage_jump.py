import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt

backend = 'numpy'
bp.backend.set(backend=backend)
brainmodels.set_backend(backend=backend)

neu1 = brainmodels.neurons.LIF(1, monitors=['V'])
neu2 = brainmodels.neurons.LIF(1, monitors=['V'])
syn = brainmodels.synapses.Voltage_jump(weight=15., pre=neu1, post=neu2, conn=bp.connect.All2All(),
                                        post_refractory=True,
                                        delay=1., monitors=['s'])

net = bp.Network(neu1, neu2, syn)
net.run(100., inputs=(neu1, 'input', 30.), report=True)

fig, gs = bp.visualize.get_figure(row_num=2, col_num=1, )
fig.add_subplot(gs[0, 0])
plt.plot(net.ts, neu1.mon.V[:, 0], label='V1')
plt.plot(net.ts, neu2.mon.V[:, 0], label='V2')
plt.legend()
fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(syn.mon.ts, syn.mon.s, show=True)
