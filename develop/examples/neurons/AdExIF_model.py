import brainpy as bp
import numpy as np
import bpmodels
import matplotlib.pyplot as plt

bp.backend.set(backend='numba', dt=.01)

duration = 200
I_ext = 65
group = bpmodels.AdExIF(size = 1, monitors = ['V'], 
                        a=.5, b=7, R=.5, tau=9.9, tau_w=100,
                        V_reset=-70, V_rest=-70, V_th=-30, 
                        V_T=-50, delta_T=2.)

group.run(duration, inputs=('input', I_ext))
bp.visualize.line_plot(group.mon.ts, group.mon.V, ylim=(-70., -35.),  show=True)
