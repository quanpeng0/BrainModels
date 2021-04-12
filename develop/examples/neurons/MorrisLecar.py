# -*- coding: utf-8 -*-

import brainpy as bp
import brainmodels

bp.backend.set(dt=0.02)
neu = brainmodels.neurons.MorrisLecar(100, monitors=['V', 'W'])

'''The current is constant'''
current = bp.inputs.ramp_current(90, 90, 1000, 0, 1000)
neu.run(duration=1000., inputs=['input', current], report=False)

fig, gs = bp.visualize.get_figure(2, 2, 3, 6)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(
    neu.mon.V[:, 0], neu.mon.W[:, 0], xlabel='Membrane potential V(mV)',
    ylabel='Recovery Variable', title='W - V')

fig.add_subplot(gs[0, 1])
plt.plot(neu.mon.ts, neu.mon.V[:, 0], label='V')
bp.visualize.line_plot(
    neu.mon.ts, neu.mon.V[:, 0], xlabel='Time (ms)',
    ylabel='Membrane potential V(mV)', title='V - t')

fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(
    neu.mon.ts, neu.mon.W[:, 0], xlabel='Time (ms)',
    ylabel='Recovery Variable', title='W - t')

fig.add_subplot(gs[1, 1])
bp.visualize.line_plot(
    neu.mon.ts, current, xlabel='Time (ms)', ylabel='Input',
    title='Input - t', show=True)
