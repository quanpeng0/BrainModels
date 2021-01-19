# -*- coding: utf-8 -*-

"""
Implementation of the paper：

- Wong, K.-F. & Wang, X.-J. A Recurrent Network Mechanism 
  of Time Integration in Perceptual Decisions. 
  J. Neurosci. 26, 1314–1328 (2006).

"""

import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt

print("version：", bp.__version__)

from collections import OrderedDict

def decision_model(coherence=0, tau_s=100., gamma=0.641,
                J_rec = .1561, J_inh = .0264,
                J_A11 = 9.9026e-04, J_A12 = 6.5177e-05, 
                J_Aext = 0.2243e-03,
                I_0=0.2346, noise = 1.,
                tau_A = 2.):
    '''
    Args:
        coherence (float): coherence level (from 0. to 1. => 0% to 100%).
        tau_s (float): NMDAR time constant.
        gamma (float): NMDAR dynamics parameters.
        J_rec (float): from self to self, NMDA; J_N11=J_N22
        J_inh (float): from others to self, NMDA; J_N21=J_N12
        J_A11 (float): from 1 to 1, AMPAR; J_A22=J_A11
        J_A12 (float): from 2 to 1, AMPAR; J_A21=J_A12
        J_Aext (float): average synaptic coupling with AMPARs.
        I_0 (float): mean effective external input.
        noise (flaot): variance of the noise.
        ----------
        tau_A (float): AMPAR time constant.
    '''
    ST = bp.types.NeuState('s1', 's2', 'r1','r2', 'input')

    def theta(x):
        if x < 0:
            return 0
        else:
            return 1

    def H(x1, x2):
        a = 239400 * J_A11 + 270
        b = 97000 * J_A11 + 108
        d = -30 * J_A11 + 0.1540

        f = -J_A12 * (-276 * x2 + 106) * theta(x2 - 0.4)
        # h1 = 507.068244 * x1 - f - 204.05522
        h1 = a * x1 - f - b
        h = h1 / (1 - np.exp(-d * h1))
        # h = h1 / (1 - np.exp(-0.1242922 * h1))
        return h

    @bp.integrate
    def int_s1(s1, t, r1):
        f = - s1 / tau_s + (1 - s1) * gamma * r1
        return f, noise / tau_s

    @bp.integrate
    def int_s2(s2, t, r2):
        f = - s2 / tau_s + (1 - s2) * gamma * r2
        return f, noise / tau_s

    def update(ST, _t):
        s1 = ST['s1']
        s2 = ST['s2']
        I1 = J_Aext * (1 + coherence) * ST['input']
        I2 = J_Aext * (1 - coherence) * ST['input']
        x1 = J_rec * s1 - J_inh * s2 + I_0 + I1
        x2 = J_rec * s2 - J_inh * s1 + I_0 + I2
        r1 = H(x1, x2)
        r2 = H(x2, x1)
        s1= int_s1(ST['s1'], _t, r1)
        ST['s2'] = int_s2(ST['s2'], _t, r2)
        ST['s1'] = s1
        ST['r1'], ST['r2'] = r1, r2
        ST['input'] = 0.

    

    return bp.NeuType(name='neuron', 
                    ST=ST, 
                    steps=update, 
                    mode='scalar')


def decision_model2(coherence=0, tau_s=100., gamma=0.641,
                J_rec = .1561, J_inh = .0264,
                J_A11 = 9.9026e-04, J_A12 = 6.5177e-05, 
                J_Aext = 0.2243e-03,
                I_0=0.2346, noise = 1.,
                tau_A = 2.):
    '''
    Args:
        coherence (float): coherence level (from 0. to 1. => 0% to 100%).
        tau_s (float): NMDAR time constant.
        gamma (float): NMDAR dynamics parameters.
        J_rec (float): from self to self, NMDA; J_N11=J_N22
        J_inh (float): from others to self, NMDA; J_N21=J_N12
        J_A11 (float): from 1 to 1, AMPAR; J_A22=J_A11
        J_A12 (float): from 2 to 1, AMPAR; J_A21=J_A12
        J_Aext (float): average synaptic coupling with AMPARs.
        I_0 (float): mean effective external input.
        noise (flaot): variance of the noise.
        ----------
        tau_A (float): AMPAR time constant.
    '''
    ST = bp.types.NeuState('s1', 's2', 'r1', 'r2', 'input')

    def theta(x):
        return np.where(x< 0, 0, 1)

    @bp.integrate
    def int_s1(s1, t, s2, I_ext):
        I1 = J_Aext * (1 + coherence) * I_ext
        I2 = J_Aext * (1 - coherence) * I_ext
        x1 = J_rec * s1 - J_inh * s2 + I_0 + I1
        x2 = J_rec * s2 - J_inh * s1 + I_0 + I2
        # H(x1, x2)
        a = 239400 * J_A11 + 270
        b = 97000 * J_A11 + 108
        d = -30 * J_A11 + 0.1540
        f = J_A12 * (-276 * x2 + 106) * theta(x2 - 0.4)
        h1 = a * x1 - f - b
        r1 = h1 / (1 - np.exp(-d * h1))
        dsdt = - s1 / tau_s + (1 - s1) * gamma * r1
        return (dsdt, noise / tau_s), r1

    @bp.integrate
    def int_s2(s2, t, s1, I_ext):
        I1 = J_Aext * (1 + coherence) * I_ext
        I2 = J_Aext * (1 - coherence) * I_ext
        x1 = J_rec * s1 - J_inh * s2 + I_0 + I1
        x2 = J_rec * s2 - J_inh * s1 + I_0 + I2
        # H(x2, x1)
        a = 239400 * J_A11 + 270
        b = 97000 * J_A11 + 108
        d = -30 * J_A11 + 0.1540
        f = J_A12 * (-276 * x1 + 106) * theta(x1 - 0.4)
        h1 = a * x2 - f - b
        r2 = h1 / (1 - np.exp(-d * h1))
        dsdt = - s2 / tau_s + (1 - s2) * gamma * r2
        return (dsdt, noise / tau_s), r2

    def update(ST, _t):
        s1, ST['r1'] = int_s1(ST['s1'], _t, ST['s2'], ST['input'])
        ST['s2'], ST['r2'] = int_s2(ST['s2'], _t, ST['s1'], ST['input'])
        ST['s1'] = s1
        ST['input'] = 0.

    return bp.NeuType(name='neuron', 
                    ST=ST, 
                    steps=update, 
                    mode='scalar')


bp.profile.set(jit=False, dt=.1
                #numerical_method='exponential'
                )

decision_th = 15        # decision threshold (Hz).
coherence = .512               # coherence level (from 0. to 1. => 0% to 100%)


model = decision_model(coherence=coherence)
neu = bp.NeuGroup(model, 1, monitors=['r1', 'r2'])
neu.run(duration=400., inputs=('ST.input', 30.), report=True)

fig, gs = bp.visualize.get_figure(1, 1, 3, 8)

fig.add_subplot(gs[0, 0])

plt.plot(neu.mon.ts, neu.mon.r1[:,0], 'r', label = 'group1')
plt.plot(neu.mon.ts, neu.mon.r2[:,0], 'b', label = 'group2')
#plt.xlim(neu.mon.t_start - 0.1, neu.mon.t_end + 0.1)
plt.ylabel('r')
plt.legend()
plt.show()


model = decision_model2(coherence=coherence)
neu = bp.NeuGroup(model, 1, monitors=['r1', 'r2'])
neu.run(duration=400., inputs=('ST.input', 0.), report=True)

fig, gs = bp.visualize.get_figure(1, 1, 3, 8)

fig.add_subplot(gs[0, 0])

plt.plot(neu.mon.ts, neu.mon.r1[:,0], 'r', label = 'group1')
plt.plot(neu.mon.ts, neu.mon.r2[:,0], 'b', label = 'group2')
#plt.xlim(neu.mon.t_start - 0.1, neu.mon.t_end + 0.1)
plt.ylabel('r')
plt.legend()
plt.show()

# phase plane analysis
'''
analyzer = bp.PhasePortraitAnalyzer(
    model=decision_model(),
    target_vars=OrderedDict(s1=[0., 1.], s2=[0., 1.]),
    fixed_vars={'r1': .003, 'r2': .003})
'''

analyzer = bp.PhasePortraitAnalyzer(
    model=decision_model2(coherence=coherence),
    target_vars=OrderedDict(s1=[0., 1.], s2=[0., 1.]),
    fixed_vars={'input': 0., 'I_ext': 0.})

#analyzer.plot_nullcline()
analyzer.plot_vector_field(show=True)
#analyzer.plot_fixed_point()
