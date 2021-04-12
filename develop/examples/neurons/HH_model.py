# -*- coding: utf-8 -*-


import brainpy as bp
import brainmodels

backend = 'numba'
bp.backend.set(backend=backend, dt=.01)
brainmodels.set_backend(backend=backend)

group = brainmodels.neurons.HH(100, monitors=['V'])

group.run(200., inputs=('input', 10.), report=True)
bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

group.run(200., report=True)
bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)
