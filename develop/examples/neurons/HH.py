# -*- coding: utf-8 -*-


import brainpy as bp
import brainmodels.numba_backend as bpmodels

bp.backend.set('numba', dt=0.01)

group = bpmodels.neurons.HH(100, monitors=['V'], show_code=True)

group.run(200., inputs=('input', 10.), report=True)
bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

group.run(200., report=True)
bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

