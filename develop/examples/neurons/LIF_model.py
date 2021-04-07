# -*- coding: utf-8 -*-
import brainpy as bp
import brainmodels

dt = 0.1
bp.backend.set('numpy', dt=dt)
neu = brainmodels.neurons.LIF(100, monitors=['V', 'refractory', 'spike'])
neu.t_refractory = 5.
net = bp.Network(neu)
net.run(duration=200., inputs=(neu, 'input', 21.), report=True)
fig, gs = bp.visualize.get_figure(3, 1, 4, 10)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(neu.mon.ts, neu.mon.V,
                       xlabel="t", ylabel="V")
