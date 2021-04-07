# -*- coding: utf-8 -*-

"""
Implementation of the paper：

- Wong, K.-F. & Wang, X.-J. A Recurrent Network Mechanism 
  of Time Integration in Perceptual Decisions. 
  J. Neurosci. 26, 1314–1328 (2006).

We adopt a simplified version from 
- Stanford University, BIOE 332: Large-Scale Neural Modeling, 
    Kwabena Boahen & Tatiana Engel, 2013, online available.

"""

import brainpy as bp
from numba import prange
from collections import OrderedDict

bp.backend.set(backend='numpy', dt=0.05)


class Decision(bp.NeuGroup):
    target_backend = 'general'

    @staticmethod
    def derivative(s1, s2, t, I, coh, JAext, J_rec, J_inh, I_0, b, d, a, tau_s, gamma):
        I1 = JAext * I * (1. + coh)
        I2 = JAext * I * (1. - coh)

        I_syn1 = J_rec * s1 - J_inh * s2 + I_0 + I1
        r1 = (a * I_syn1 - b) / (1. - bp.backend.exp(-d * (a * I_syn1 - b)))
        ds1dt = - s1 / tau_s + (1. - s1) * gamma * r1

        I_syn2 = J_rec * s2 - J_inh * s1 + I_0 + I2
        r2 = (a * I_syn2 - b) / (1. - bp.backend.exp(-d * (a * I_syn2 - b)))
        ds2dt = - s2 / tau_s + (1. - s2) * gamma * r2

        return ds1dt, ds2dt

    def __init__(self, size, coh, tau_s=.06, gamma=0.641,
                 J_rec=.3725, J_inh=.1137,
                 I_0=.3297, JAext=.00117,
                 a=270., b=108., d=0.154,
                 **kwargs):
        # parameters
        self.coh = coh
        self.tau_s = tau_s
        self.gamma = gamma
        self.J_rec = J_rec
        self.J_inh = J_inh
        self.I0 = I_0
        self.JAext = JAext
        self.a = a
        self.b = b
        self.d = d

        # variables
        self.s1 = bp.backend.ones(size) * .06
        self.s2 = bp.backend.ones(size) * .06
        self.input = bp.backend.zeros(size)

        self.integral = bp.odeint(f=self.derivative, method='rk4', dt=0.01)

        super(Decision, self).__init__(size=size, **kwargs)

    def update(self, _t):
        for i in prange(self.size):
            self.s1[i], self.s2[i] = self.integral(self.s1[i], self.s2[i], _t,
                                                   self.input[i], self.coh, self.JAext, self.J_rec,
                                                   self.J_inh, self.I0, self.b, self.d,
                                                   self.a, self.tau_s, self.gamma)
            self.input[i] = 0.


if __name__ == "__main__":
    # phase plane analysis
    pars = dict(tau_s=.06, gamma=0.641,
                J_rec=.3725, J_inh=.1137,
                I_0=.3297, JAext=.00117,
                b=108., d=0.154, a=270.)

    pars['I'] = 30.
    pars['coh'] = .512

    decision = Decision(1, coh=pars['coh'])

    phase = bp.analysis.PhasePlane(decision.integral,
                                   target_vars=OrderedDict(s2=[0., 1.], s1=[0., 1.]),
                                   fixed_vars=None,
                                   pars_update=pars,
                                   numerical_resolution=.001,
                                   options={'escape_sympy_solver': True})

    phase.plot_nullcline()
    phase.plot_fixed_point()
    phase.plot_vector_field(show=True)
