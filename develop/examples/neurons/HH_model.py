# -*- coding: utf-8 -*-
import brainpy as bp
import brainmodels

backend = 'numba'
bp.backend.set(backend=backend, dt=.01)
brainmodels.set_backend(backend=backend)
neu = brainmodels.neurons.HH(100, monitors=['V'])
net = bp.Network(neu)
net.run(200., inputs=('input', 10.), report=True)

bp.visualize.line_plot(neu.mon.ts, neu.mon.V, show=True)
neu.run(200., report=True)
bp.visualize.line_plot(neu.mon.ts, neu.mon.V, show=True)
