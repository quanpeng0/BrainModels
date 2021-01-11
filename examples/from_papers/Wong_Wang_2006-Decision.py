# -*- coding: utf-8 -*-

"""
Implementation of the paper：

- 

"""

import brainpy as bp
import numpy as np

# -------
# neuron
# -------

def rate_neuron(I_0=0.2346, tau_A = 2, sigma = 0.007):
    '''
    Args:
        I_0 (float): mean effective external input.
        tau_A (float): AMPAR time constant.
        sigma (flaot): variance of the noise.
    '''
    ST = bp.types.NeuState(['r', 'input', 'I_noise'])

    @bp.integrate
    def int_I_noise(I_noise, t):
        eta = np.random.randn(len(I_noise))     # Gassian white noise
        return -I_noise + eta * np.sqrt(tau_A * (sigma ** 2))

    def get_x(I_noise, I_ext):
        return I_noise + I_0 + I_ext

    def H(x1, x2):
        return x1+x2  # ??

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
    ST = bp.types.SynState('s', 'g')
    pre = bp.types.NeuState('r')
    post = bp.types.NeuState('r')

    @bp.integrate
    def int_s(s, t, r):
        return - s / tau + (1 - s) * gamma * r

    def update(ST, _t, pre):
        ST['s'] = int_s(ST['s'], _t, pre['r'])
        ST['g'] = J * ST['s']

    def output(ST, post):
        post['r'] += ST['g']
        
    return bp.SynType(name='NMDA_synapse',
                      ST=ST,
                      steps=(update, output),
                      requires=dict(pre=pre, post=post),
                      mode='scalar')


# -------
# network
# -------


def decision_net(group, syn, J_rec = .1561, J_inh = .0264):
    rec_conn = bp.SynConn(syn,
                        pre_group=group,
                        post_group=group,
                        conn = bp.connect.One2One()
                        )

    rec_conn.pars['J'] = J_rec

    inh_conn = bp.SynConn(syn,
                        pre_group=group,
                        post_group=group,
                        conn = bp.connect.All2All(include_self=False))

    inh_conn.pars['J'] = -J_inh

    return bp.Network(group, rec_conn, inh_conn)


# -------
# simulation
# -------

bp.profile.set(jit=True, numerical_method='exponential')

decision_th = 15    # decision threshold (Hz)

# build network
J_rec = .1561
J_inh = .0264
neu = rate_neuron()
syn = get_syn()
group = bp.NeuGroup(neu, geometry=2,
                    monitors=['r'])
rec_conn = bp.SynConn(syn,
                    pre_group=group,
                    post_group=group,
                    conn = bp.connect.One2One()
                    )

rec_conn.pars['J'] = J_rec

inh_conn21 = bp.SynConn(syn,
                    pre_group=group[:1],
                    post_group=group[1:],
                    conn = bp.connect.All2All())

inh_conn21.pars['J'] = -J_inh

inh_conn12 = bp.SynConn(syn,
                    pre_group=group[1:],
                    post_group=group[:1],
                    conn = bp.connect.All2All())

inh_conn12.pars['J'] = -J_inh

net = bp.Network(group, rec_conn, inh_conn21, inh_conn12)

# simulation
amplitude = 100.
J_Aext = 0.2243*(10**(-3))      # average synaptic coupling with AMPARs
coherence = .512                # coherence level (from 0. to 1. => 0% to 100%)
# co = 0
I1 = J_Aext * (1 + coherence) * amplitude 
I2 = J_Aext * (1 - coherence) * amplitude 

I = np.array([I1, I2]).T
print(I)

net.run(duration=100., inputs=(group, 'ST.input', I), report=True)



'''
# J_Aij: from j to i, AMPAR
J_A11 = 9.9026*(10**(-4))
J_A22 = J_A11
J_A12 = 6.5177*(10**(-5))
J_A21 = J_A12
'''