import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt

print(bp.__version__)

bp.backend.set(backend='numpy', dt=.05)

neu = brainmodels.neurons.LIF(1, V_rest=-65., V_reset=-65., V_th=-55.)
neu.V = -65. * bp.ops.ones(neu.V.shape)

(I, duration) = bp.inputs.constant_current([(0, 1), (30, 8), (0, 300)])

# NMDA
syn = brainmodels.synapses.NMDA(monitors=['s'], pre=neu, post=neu, conn=bp.connect.All2All())
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(neu, "input", I))
post_input = syn.mon.s[:, 0] * 0.15 * (-65-0) * -50
plt.plot(net.ts, post_input, label='NMDA')

# AMPA
neu = brainmodels.neurons.LIF(1, V_rest=-65., V_reset=-65., V_th=-55.)
neu.V = -65. * bp.ops.ones(neu.V.shape)
syn = brainmodels.synapses.AMPA1(monitors=['s'], pre=neu, post=neu, conn=bp.connect.All2All())
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(neu, "input", I))
post_input = syn.mon.s[:, 0] * 0.1 * (-65-0) * -50
plt.plot(net.ts, post_input, label='AMPA')

# GABA_b
neu = brainmodels.neurons.LIF(1, V_rest=-65., V_reset=-65., V_th=-55.)
neu.V = -65. * bp.ops.ones(neu.V.shape)
syn = brainmodels.synapses.GABAb1(T_duration=0.005, monitors=['s'], pre=neu, post=neu, conn=bp.connect.All2All())
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(neu, "input", I))
post_input = syn.mon.s[:, 0] * (-65+95) * -300000 * 1e+7
plt.plot(net.ts, post_input, label='GABAb')

# GABA_a
neu = brainmodels.neurons.LIF(1, V_rest=-65., V_reset=-65., V_th=-55.)
neu.V = -65. * bp.ops.ones(neu.V.shape)
syn = brainmodels.synapses.GABAa1(tau=2, monitors=['s'], pre=neu, post=neu, conn=bp.connect.All2All())
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(neu, "input", I))
post_input = syn.mon.s[:, 0] * 0.4 * (-65+80) * -50
plt.plot(net.ts, post_input, label='GABAa')

# general plot settings
plt.ylabel('-I')
plt.xlabel('Time (ms)')
plt.legend()
plt.show()
