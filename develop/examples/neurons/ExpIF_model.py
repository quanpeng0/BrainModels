# -*- coding: utf-8 -*-
import brainpy as bp
import brainmodels

neu = brainmodels.neurons.ExpIF(16, monitors=['V', 'spike', 'refractory'])

neu.run(duration = 50., inputs=('input', 0.23), report=True)
bp.visualize.line_plot(neu.mon.ts, neu.mon.V, 
                       xlabel = "t", ylabel = "V", 
                       show=True)

