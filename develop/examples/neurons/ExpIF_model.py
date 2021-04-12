# -*- coding: utf-8 -*-
import brainpy as bp
import brainmodels

neu = brainmodels.numba_backend.neurons.ExpIF(16, monitors=['V', 'spike', 'refractory'])

neu.run(duration=100, inputs=('input', 1.), report=True)
bp.visualize.line_plot(neu.mon.ts, neu.mon.V,
                       xlabel="t", ylabel="V",
                       show=True)
