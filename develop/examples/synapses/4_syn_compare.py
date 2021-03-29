import numpy as np
import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt

print(bp.__version__)

bp.backend.set(backend='numpy', dt=.05)

neu = brainmodels.neurons.LIF(1, V_rest=-65., V_reset=-65., V_th=-55.)
neu.V = -65. * np.ones(neu.V.shape)

(I, duration) = bp.inputs.constant_current([(0, 1), (30, 5), (0, 500)])

# NMDA
syn = brainmodels.synapses.NMDA(g_max=0.03, monitors=['s'], pre=neu, post=neu, conn=bp.connect.All2All())
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(neu, "input", I))
plt.plot(net.ts, 5000 * syn.mon.s[:, 0], label='NMDA')

# AMPA
syn = brainmodels.synapses.AMPA1(g_max=0.001, monitors=['s'], pre=neu, post=neu, conn=bp.connect.All2All())
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(neu, "input", I))
plt.plot(net.ts, 5000 * syn.mon.s[:, 0], label='AMPA')

# GABA_b
syn = brainmodels.synapses.GABAb1(T_duration=0.15, monitors=['s'], pre=neu, post=neu, conn=bp.connect.All2All())
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(neu, "input", I))
plt.plot(net.ts, 1e+6 * 5000 * syn.mon.s[:, 0], label='GABAb')

# GABA_a
syn = brainmodels.synapses.GABAa1(g_max=0.002, monitors=['s'], pre=neu, post=neu, conn=bp.connect.All2All())
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(neu, "input", I))
plt.plot(net.ts, 5000 * syn.mon.s[:, 0], label='GABAa')


plt.ylabel('-I')
plt.xlabel('Time (ms)')
plt.legend()
plt.show()