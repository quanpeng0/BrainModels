# -*- coding: utf-8 -*-

import brainpy as bp
import brainmodels

mode = 'irregular_bursting'
param= {'quiescence':         [1.0, 2.0],  #a
        'spiking':            [3.5, 5.0],  #c
        'bursting':           [2.5, 3.0],  #d
        'irregular_spiking':  [2.95, 3.3], #h
        'irregular_bursting': [2.8, 3.7],  #g
        }  
#set params of b and I_ext corresponding to different firing mode
print(f"parameters is set to firing mode <{mode}>")

group = brainmodels.neurons.HindmarshRose(size = 10, b = param[mode][0],
                      monitors=['V'])

group.run(100., inputs=('input', param[mode][1]), report=True)
bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

# phase plane analysis
phase = bp.analysis.PhasePlane(
    brainmodels.neurons.HindmarshRose.integral,
    target_vars={'V': [-3, 3], 'y': [-20., 5.]},
    fixed_vars={'z': 0.},
    pars_update={'I_ext': param[mode][1], 
                 'a': 1., 'b': 3., 'c': 1., 'd': 5., 
                 'r': 0.01, 's': 4., 'V_rest': -1.6})
phase.plot_nullcline()
phase.plot_fixed_point()
phase.plot_vector_field()
phase.plot_trajectory([{'V': 1., 'y': 0., 'z':-0.0}],
                      duration=100.,
                      show=True)
