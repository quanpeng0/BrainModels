# -*- coding: utf-8 -*-
import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt
import numpy as np

mode2par = {
    'tonic': [0.02, 0.40, -65.0, 2.0],
    'tonic spiking': [0.02, 0.40, -65.0, 2.0],
    'phasic': [0.02, 0.25, -65.0, 6.0],
    'phasic spiking': [0.02, 0.25, -65.0, 6.0],
    'tonic bursting': [0.02, 0.20, -50.0, 2.0],
    'phasic bursting': [0.02, 0.25, -55.0, 0.05],
    'mixed mode': [0.02, 0.20, -55.0, 4.0],
    'SFA': [0.01, 0.20, -65.0, 8.0],
    'spike frequency adaptation': [0.01, 0.20, -65.0, 8.0],
    'class 1': [0.02, -0.1, -55.0, 6.0],
    'class 2': [0.20, 0.26, -65.0, 0.0],
    'spike latency': [0.02, 0.20, -65.0, 6.0],
    'subthreshold oscillation': [0.05, 0.26, -60.0, 0.0],
    'resonator': [0.10, 0.26, -60.0, -1.0],
    'integrator': [0.02, -0.1, -55.0, 6.0],
    'rebound spike': [0.03, 0.25, -60.0, 4.0],
    'rebound burst': [0.03, 0.25, -52.0, 0.0],
    'threshold variability': [0.03, 0.25, -60.0, 4.0],
    'bistability': [1.00, 1.50, -60.0, 0.0],
    'DAP': [1.00, 0.20, -60.0, -21.0],
    'depolarizing afterpotential': [1.00, 0.20, -60.0, -21.0],
    'accommodation': [0.02, 1.00, -55.0, 4.0],
    'inhibition-induced spiking': [-0.02, -1.00, -60.0, 8.0],
    'inhibition-induced bursting': [-0.026, -1.00, -45.0, 0],

    'Regular Spiking': [0.02, 0.2, -65, 8],
    'RS': [0.02, 0.2, -65, 8],
    'Intrinsically Bursting': [0.02, 0.2, -55, 4],
    'IB': [0.02, 0.2, -55, 4],
    'Chattering': [0.02, 0.2, -50, 2],
    'CH': [0.02, 0.2, -50, 2],
    'Fast Spiking': [0.1, 0.2, -65, 2],
    'FS': [0.1, 0.2, -65, 2],
    'Thalamo-cortical': [0.02, 0.25, -65, 0.05],
    'TC': [0.02, 0.25, -65, 0.05],
    'Resonator': [0.1, 0.26, -65, 2],
    'RZ': [0.1, 0.26, -65, 2],
    'Low-threshold Spiking': [0.02, 0.25, -65, 2],
    'LTS': [0.02, 0.25, -65, 2]
}

bp.backend.set("numpy")

mode = 'Regular Spiking'
neu = brainmodels.neurons.Izhikevich(10, monitors=['V', 'u'])
neu.a = mode2par[mode][0]
neu.b = mode2par[mode][1]
neu.c = mode2par[mode][2]
neu.d = mode2par[mode][3]

current2 = bp.inputs.ramp_current(10, 10, 300, 0, 300)
current1 = np.zeros(int(np.ceil(100 / 0.1)))
current = np.append(current1, current2)
neu.run(duration=400., inputs=['input', current], report=False)

fig, gs = bp.visualize.get_figure(3, 1, 3, 8)

fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(neu.mon.ts, neu.mon.V[:, 0], xlabel='Time (ms)',
                       ylabel='Membrane potential', xlim=[-0.1, 400.1])
fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(
    neu.mon.ts, current, xlim=[-0.1, 400.1], ylim=[0, 60],
    xlabel='Time (ms)', ylabel='Input(mV)'
)
fig.add_subplot(gs[2, 0])
bp.visualize.line_plot(
    neu.mon.ts, neu.mon.u[:, 0], xlim=[-0.1, 400.1],
    xlabel='Time (ms)', ylabel='Recovery variable', show=True)

mode = 'tonic spiking'
neu = brainmodels.neurons.Izhikevich(10, monitors=['V', 'u'])
neu.a = mode2par[mode][0]
neu.b = mode2par[mode][1]
neu.c = mode2par[mode][2]
neu.d = mode2par[mode][3]

current2 = bp.inputs.ramp_current(10, 10, 150, 0, 150)
current1 = np.zeros(int(np.ceil(50 / 0.1)))
current = np.append(current1, current2)
neu.run(duration=200., inputs=['input', current], report=False)

fig, ax1 = plt.subplots(figsize=(15, 5))
plt.title('Tonic Spiking')
ax1.plot(neu.mon.ts, neu.mon.V[:, 0], label='V')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Membrane potential (mV)')
ax1.set_xlim(-0.1, 200.1)
ax1.tick_params('y')
ax2 = ax1.twinx()
ax2.plot(neu.mon.ts, current, 'c', label='Input')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Input (mV)')
ax2.set_ylim(0, 50)
ax2.tick_params('y')
ax1.legend(loc=1)
ax2.legend(loc=3)
fig.tight_layout()
plt.show()
