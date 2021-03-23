import brainpy as bp
import bpmodels
import matplotlib.pyplot as plt
import numpy as np

print(bp.__version__)

neu0 = bpmodels.neurons.LIF(1, monitors=['V'])
neu1 = bpmodels.neurons.LIF(1, monitors=['V'])

# gap junction for lif
syn = bpmodels.Gap_junction_lif(pre=neu0, post=neu1, conn=bp.connect.All2All(),
                    k_spikelet=0.1)

net = bp.Network(neu0, neu1, syn)
net.run(100., inputs=(neu0, 'input', 30.), report=True)

plt.plot(net.ts, neu0.mon.V[:, 0], label='V0')
plt.plot(net.ts, neu1.mon.V[:, 0], label='V1')
plt.legend()
plt.show()