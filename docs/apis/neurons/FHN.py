# -*- coding: utf-8 -*-
# %% [markdown]
# # Example of FitzHugh–Nagumo model
# %%
import brainpy as bp
import brainmodels


# %%
# simulation
fnh = brainmodels.neurons.FHN(1, monitors=['V', 'w'])
fnh.run(100., inputs=('input', 1.), report=0.1)
bp.visualize.line_plot(fnh.mon.ts, fnh.mon.w, legend='w')
bp.visualize.line_plot(fnh.mon.ts, fnh.mon.V, legend='V', show=True)

# %%
# phase plane analysis
phase = bp.symbolic.PhasePlane(
  # fnh, target_vars={'V': [-2, 2], 'w': [-0.5, 2.5]},
  fnh, target_vars={'V': [-3, 3], 'w': [-1, 3]},
  pars_update={'Iext': 1., 'a': 0.7, 'b': 0.8, 'tau': 12.5})
phase.plot_nullcline()
phase.plot_fixed_point()
phase.plot_limit_cycle_by_sim(initials={'V': -1, 'w': 1}, duration=100.)
phase.plot_vector_field(show=True)

# %%
# bifurcation analysis
bifurcation = bp.symbolic.Bifurcation(
  fnh, target_pars=dict(a=[0.3, 0.8], Iext=[-1, 1], ),
  target_vars={'V': [-3, 2], 'w': [-2, 2]},
  pars_update={'b': 0.8, 'tau': 12.5},
  numerical_resolution=0.01)
_ = bifurcation.plot_bifurcation(show=True)
