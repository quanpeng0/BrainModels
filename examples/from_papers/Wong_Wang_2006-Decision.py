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

bp.profile.set(jit=False, numerical_method='exponential')

# -------
# neuron
# -------

def get_neuron(I_0=0.2346, tau_A = 2, sigma = 0.007, decision_th = 15, J_A11 = 9.9026, J_A12 = 6.5177*(10**(-5))):
    '''
    Args:
        I_0 (float): mean effective external input.
        tau_A (float): AMPAR time constant.
        sigma (flaot): variance of the noise.
        decision_th (float): decision threshold (Hz).
        J_A11 (float): from 1 to 1, AMPAR; J_A22=J_A11
        J_A12 (float): from 2 to 1, AMPAR; J_A21=J_A12
    '''
    ST = bp.types.NeuState('r', 'input', 'I_noise')

    @bp.integrate
    def int_I_noise(I_noise, t):
        eta = np.random.randn(len(I_noise))     # Gassian white noise
        return -I_noise + eta * np.sqrt(tau_A * (sigma ** 2))

    def get_x(I_noise, I_ext):
        return I_noise + I_0 + I_ext

    def theta(x):
        if x < 0:
            return 0
        else:
            return 1

    def H(x1, x2):
        a = 23.9400 * J_A11 + 270
        b = 9.7000 * J_A11 + 108
        d = -.003 * J_A11 + 0.1540
        f = J_A12 * (-276 * x2 + 106) * theta(x2 - 0.4)
        h1 = a * x1 - f - b
        h = h1 / (1 - np.exp(-d * h1))
        return h

    def get_r(x1, x2):
        r1 = H(x1, x2)
        r2 = H(x2, x1)
        return np.array([r1, r2])

    def update(ST, _t):
        I_noise = int_I_noise(ST['I_noise'], _t)
        x1 = get_x(I_noise[0], ST['input'][0])
        x2 = get_x(I_noise[1], ST['input'][1])
        ST['r'] = get_r(x1, x2)
        ST['input'] = np.zeros(len(ST['input']))

    return bp.NeuType(name='neuron', 
                    ST=ST, 
                    steps=update, 
                    mode='vector')


# -------
# synapse
# -------

def get_syn(J=.1, tau=100, gamma=0.641):
    '''
    Args:
        J (float): effective coupling constants from pre to post by NMDAR.
        tau (float): NMDAR time constant.
        gamma (float):
    '''
    ST = bp.types.SynState('s')
    pre = bp.types.NeuState('r')
    post = bp.types.NeuState('input')

    @bp.integrate
    def int_s(s, t, r):
        return - s / tau + (1 - s) * gamma * r

    def update(ST, _t, pre):
        ST['s'] = int_s(ST['s'], _t, pre['r'])

    def output(ST, post):
        post['input'] += J * ST['s']
        
    return bp.SynType(name='NMDA_synapse',
                      ST=ST,
                      steps=(update, output),
                      requires=dict(pre=pre, post=post),
                      mode='scalar')


# -------
# network
# -------

J_rec = .1561
J_inh = -.0264
neu = get_neuron()
syn = get_syn()
group = bp.NeuGroup(neu, geometry=2, monitors=['r'])
rec_conn = bp.SynConn(syn,
                    pre_group=group,
                    post_group=group,
                    conn = bp.connect.One2One())

rec_conn.pars['J'] = J_rec

inh_conn = bp.SynConn(syn,
                    pre_group=group,
                    post_group=group,
                    conn = bp.connect.All2All(include_self=False))

inh_conn.pars['J'] = J_inh

net = bp.Network(group, rec_conn, inh_conn)

# simulation
amplitude = 10.
J_Aext = 0.2243*(10**(-3))      # average synaptic coupling with AMPARs
coherence = .512                # coherence level (from 0. to 1. => 0% to 100%)
#coherence = 0
I1 = J_Aext * (1 + coherence) * amplitude 
I2 = J_Aext * (1 - coherence) * amplitude 

def get_current(amplitude, duration=2000):
    (I, duration) = bp.inputs.constant_current(
                            [(0, 500), (amplitude, duration-500)])
    return I

I = np.array([get_current(I1), get_current(I2)]).T

net.run(duration=2000., inputs=(group, 'ST.input', I), report=False)

fig, gs = bp.visualize.get_figure(1, 1, 3, 8)

fig.add_subplot(gs[0, 0])

plt.plot(net.ts, group.mon.r[:, 0], 'r', label = 'group1')
plt.plot(net.ts, group.mon.r[:, 1], 'b', label = 'group2')
plt.xlim(net.t_start - 0.1, net.t_end + 0.1)
plt.ylabel('Firing Rate (Hz)')
plt.legend()
plt.show()