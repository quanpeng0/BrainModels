# -*- coding: utf-8 -*-

# setup
import brainmodels
import brainpy as bp
FixedProb = bp.connect.FixedProb
pars = dict(V_rest=-60, V_th=-50, V_reset=-60, tau=20)

# neuron groups
E = brainmodels.LIF(3200, tau_ref=5, monitors=['spike'], **pars)
I = brainmodels.LIF(800, tau_ref=10, **pars)

# synapses
E2E = brainmodels.ExpCOBA(E, E, FixedProb(prob=0.02), E=0, g_max=0.6, tau=5)
E2I = brainmodels.ExpCOBA(E, I, FixedProb(prob=0.02), E=0, g_max=0.6, tau=5)
I2E = brainmodels.ExpCOBA(I, E, FixedProb(prob=0.02), E=-80, g_max=6.7, tau=10)
I2I = brainmodels.ExpCOBA(I, I, FixedProb(prob=0.02), E=-80, g_max=6.7, tau=10)

# network
net = bp.math.jit(bp.Network(E2E, E2I, I2I, I2E, E=E, I=I))

# simulation and visualization
net.run(1000., inputs=[('E.input', 20), ('I.input', 20)])
bp.visualize.raster_plot(E.mon.ts, E.mon.spike, show=True)
