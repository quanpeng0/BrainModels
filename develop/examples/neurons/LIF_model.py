# -*- coding: utf-8 -*-
import brainpy as bp
import brainmodels
    
neu = brainmodels.neurons.LIF(16, monitors=['V', 'refractory'])

neu.run(duration = 100., inputs=('input', 21.), report=True)
bp.visualize.line_plot(neu.mon.ts, neu.mon.V, 
                       xlabel = "t", ylabel = "V", 
                       show=True)
bp.visualize.line_plot(neu.mon.ts, neu.mon.refractory, 
                       xlabel = "t", ylabel = "ref", 
                       show=True)
