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

print("version：", bp.__version__)

def decision_model(coherence=0, tau_s=100., gamma=0.641,
                J_rec = .1561, J_inh = .0264,
                J_A11 = 9.9026e-04,
                I_0=0.2346, noise = .02, tau_A = 0.002):
    '''
    Args:
        coherence (float): coherence level (from 0. to 1. => 0% to 100%).
        tau_s (float): NMDAR time constant.
        gamma (float): NMDAR dynamics parameters.
        J_rec (float): from self to self, NMDA; J_N11=J_N22
        J_inh (float): from others to self, NMDA; J_N21=J_N12
        J_A11 (float): from 1 to 1, AMPAR; J_A22=J_A11
        I_0 (float): mean effective external input.
        noise (flaot): variance of the noise.
    '''
    ST = bp.types.NeuState('s1', 's2', 'r1', 'r2', 'input1','input2', 'noise1', 'noise2')

    @bp.integrate
    def int_s1(s1, t, s2, I1, I_noise1):
        x1 = J_rec * s1 - J_inh * s2 + I_0 + I1 + I_noise1
        h1 = 507.068244 * x1 - 204.05522
        r1 = h1 / (1 - np.exp(-0.1242922 * h1))
        r1[r1<0] = 0.
        dsdt = - s1 / tau_s + (1 - s1) * gamma * r1
        return (dsdt, noise / tau_s), r1

    @bp.integrate
    def int_s2(s2, t, s1, I2, I_noise2):
        x2 = J_rec * s2 - J_inh * s1 + I_0 + I2 + I_noise2
        h1 = 507.068244 * x2 - 204.05522
        r2 = h1 / (1 - np.exp(-0.1242922 * h1))
        r2[r2<0] = 0.
        dsdt = - s2 / tau_s + (1 - s2) * gamma * r2
        return (dsdt, noise / tau_s), r2

    def update(ST, _t):
        s1, ST['r1']= int_s1(ST['s1'], _t, ST['s2'], ST['input1'], ST['noise1'])
        ST['s2'], ST['r2']= int_s2(ST['s2'], _t, ST['s1'], ST['input2'], ST['noise2'])
        #ST['noise1'] = int_noise(ST['noise1'], _t)
        #ST['noise2'] = int_noise(ST['noise2'], _t)
        ST['s1'] = s1
        ST['input1'], ST['input2'] = 0., 0.

    return bp.NeuType(name='neuron', 
                    ST=ST, 
                    steps=update, 
                    mode='scalar')


def stimulus(coh,mu0,ONOFF, JAext):
    I1 = JAext * mu0 * (1 + coh) * ONOFF
    I2 = JAext * mu0 * (1 - coh) * ONOFF
    return I1, I2



bp.profile.set(jit=False, dt=.1
                #numerical_method='exponential'
                )

decision_th = 15        # decision threshold (Hz).
coherence = .512               # coherence level (from 0. to 1. => 0% to 100%)
J_Aext = 0.2243e-03
mu0 = 30.
I1, I2 = stimulus(coherence, mu0, 1., J_Aext)

model = decision_model()
neu = bp.NeuGroup(model, 1, monitors=['r1', 'r2'])
neu.run(duration=400., inputs=(['ST.input1', I1], ['ST.input2', I2]), report=True)

fig, gs = bp.visualize.get_figure(1, 1, 3, 8)

fig.add_subplot(gs[0, 0])

plt.plot(neu.mon.ts, neu.mon.r1[:,0], 'r', label = 'group1')
plt.plot(neu.mon.ts, neu.mon.r2[:,0], 'b', label = 'group2')
plt.ylabel('r')
plt.legend()
plt.show()


# phase plane analysis
analyzer = bp.PhasePortraitAnalyzer(
    model=decision_model(),
    target_vars=OrderedDict(s1=[0., 1.], s2=[0., 1.]),
    fixed_vars={'I1': I1, 'I2': I2})

analyzer.plot_nullcline()
analyzer.plot_vector_field(show=True)
analyzer.plot_fixed_point()
