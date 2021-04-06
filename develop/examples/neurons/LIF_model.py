# -*- coding: utf-8 -*-
import brainpy as bp
import brainmodels

dt = 0.02
bp.backend.set('numpy', dt=dt)
neu = brainmodels.neurons.LIF(100, monitors=['V', 'refractory', 'spike'])

net = bp.Network(neu)
net.run(duration=200., inputs=(neu, 'input', 21.), report=True)
bp.visualize.line_plot(neu.mon.ts, neu.mon.V,
                       xlabel="t", ylabel="V",
                       show=True)
