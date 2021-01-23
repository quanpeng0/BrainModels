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
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict


def decision_model(tau_s=.06, gamma=0.641,
                J_rec = .3725, J_inh = .1137, 
                I_0=.3297,
                a = 270., b=108., d=0.154):

    ST = bp.types.NeuState('r1', 'r2','I1','I2', s1=0.06, s2=0.06)
    
    @bp.integrate
    def int_s1(s1, t, s2, I1): 
        I_syn = J_rec * s1 - J_inh * s2 + I_0 + I1
        r1 = (a * I_syn - b) / (1. - np.exp(-d * (a * I_syn - b)))
        dsdt = - s1 / tau_s + (1. - s1) * gamma * r1
        return (dsdt,), r1

    @bp.integrate
    def int_s2(s2, t, s1, I2):
        I_syn = J_rec * s2 - J_inh * s1 + I_0 + I2
        r2 = (a * I_syn - b) / (1. - np.exp(-d * (a * I_syn - b)))
        dsdt = - s2 / tau_s + (1. - s2) * gamma * r2
        return (dsdt,), r2

    def update(ST, _t):
        s1, ST['r1'] = int_s1(ST['s1'], _t, ST['s2'],ST['I1'])
        s2, ST['r2'] = int_s2(ST['s2'], _t, ST['s1'],ST['I2'])
        ST['s1'] = s1
        ST['s2'] = s2
        ST['I1'], ST['I2'] = 0, 0

    return bp.NeuType(name='neuron', 
                    ST=ST, 
                    steps=update, 
                    mode='scalar')


def stimulus(coh, mu0=30., JAext=.00117):
    I1 = JAext * mu0 * (1 + coh) 
    I2 = JAext * mu0 * (1 - coh) 
    return I1, I2


I1, I2 = stimulus(coh=.512)

analyzer = bp.PhasePortraitAnalyzer(
    model=decision_model(),
    target_vars=OrderedDict(s2=[0., 1.], s1=[0., 1.]),
    fixed_vars={'I1': I1, 'I2':I2},
    options={'resolution': 0.001,
            'escape_sympy_solver': True}
)

analyzer.plot_vector_field()
analyzer.plot_nullcline()
analyzer.plot_fixed_point(show=True)