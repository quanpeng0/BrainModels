# -*- coding: utf-8 -*-
import brainpy as bp
import brainmodels

neu = brainmodels.neurons.ExpIF(16, monitors=['V'])

neu.run(duration = 200., inputs=('input', 0.3), report=True)
bp.visualize.line_plot(neu.mon.ts, neu.mon.V, 
                       xlabel = "t", ylabel = "V", 
                       show=True)
