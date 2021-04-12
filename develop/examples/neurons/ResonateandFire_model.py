# -*- coding: utf-8 -*-
import brainpy as bp
import brainmodels

bp.backend.set('numpy', dt=0.002)
group = brainmodels.neurons.ResonateandFire(1, monitors=['x', 'V'], show_code=False)
current = bp.inputs.spike_current(points=[0.0], lengths=0.002,
                                  sizes=-2., duration=20.)
group.run(duration=20., inputs=('input', current), report=True)  # is this proper input value?
bp.visualize.line_plot(group.mon.x, group.mon.V, show=True)
bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)
