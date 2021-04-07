import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt

brainmodels.set_backend(backend='numpy')

neu0 = brainmodels.neurons.LIF(2, monitors=['V'], t_refractory=0)
neu0.V = bp.backend.ones(neu0.V.shape) * -10.
neu1 = brainmodels.neurons.LIF(3, monitors=['V'], t_refractory=10)
neu1.V = bp.backend.ones(neu1.V.shape) * -10.

# gap junction for lif
syn = brainmodels.synapses.Gap_junction_lif(pre=neu0, post=neu1, conn=bp.connect.All2All(),
                                            post_refractory=True,
                                            k_spikelet=25.)
syn.w = bp.backend.ones(syn.w.shape) * .5
# syn = brainmodels.synapses.Gap_junction(pre=neu0, post=neu1, conn=bp.connect.All2All())
net = bp.Network(neu0, neu1, syn, show_code=False)
net.run(100., inputs=(neu0, 'input', 36.), report=True)

fig, gs = bp.visualize.get_figure(row_num=2, col_num=1, )

fig.add_subplot(gs[1, 0])
plt.plot(net.ts, neu0.mon.V[:, 0], label='V0')
plt.legend()

fig.add_subplot(gs[0, 0])
plt.plot(net.ts, neu1.mon.V[:, 0], label='V1')
plt.legend()
plt.show()
