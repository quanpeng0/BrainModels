# -*- coding: utf-8 -*-

"""
Implementation of the paper：

- Maass, Wolfgang; Natschläger, Thomas; Markram, Henry 
(November 2002), "Real-time computing without stable 
states: a new framework for neural computation based 
on perturbations", Neural Comput, 14 (11): 2531–60

"""

import brainpy as bp
import numpy as np
import bpmodels
import matplotlib.pyplot as plt

bp.profile.set(jit=True, device='cpu',
               numerical_method='exponential')

np.random.seed(123)

# -------
# neuron
# -------


# liquid geometry
geometry=(15, 3, 3)     # N = 135
n_x, n_y, n_z = geometry
N = int(n_x * n_y * n_z)
num_exc = int(N * .8)
num_inh = int(N * .2)
all_idx = np.arange(N)
i_idx = np.sort(np.random.choice(all_idx, num_inh, replace=False))
e_idx = np.delete(all_idx, i_idx)

# readout
N_readout = 51

# background current
I_b = 13.5    # nA

# neuron parameters
t_ref_e = 3
t_ref_i = 2
tau_m = 30    # ms
V_th = 15.    # mV
V_rest = 0.
V_reset = 13.5
R = 1.        # m

neuron = bpmodels.neurons.get_LIF(tau = tau_m, V_th = V_th, R = R,
                                    V_reset=V_reset, V_rest=V_rest)

liquid_E = bp.NeuGroup(neuron, geometry=num_exc)
liquid_I = bp.NeuGroup(neuron, geometry=num_inh)
liquid_E.ST['V'] = np.random.rand(num_exc)*(15-13.5)+13.5 # uniform distribution from [13.5 mV, 15.0 mV]
liquid_I.ST['V'] = np.random.rand(num_inh)*(15-13.5)+13.5 # uniform distribution
liquid_E.pars['t_refractory'] = t_ref_e
liquid_I.pars['t_refractory'] = t_ref_i

readout = bp.NeuGroup(neuron, geometry=N_readout)



# --------------
# liquid synapse
# --------------

lbd = 1.2*5    # controls both the average number of connections and the average distance between connected neurons
lbd_input = 3.3*5     #???

# parameters
pars_ee = dict(C = .3, U = .5, taud = 1100, tauf = 50, A = 30)
pars_ei = dict(C = .2, U = .05, taud = 125, tauf = 1200, A = 60)
pars_ie = dict(C = .4, U = .25, taud = 700, tauf = 20, A = -19)
pars_ii = dict(C = .1, U = .32, taud = 144, tauf = 60, A = -19)
# mean scaling parameter A (in nA) (w???)
# A = {'II': [2.8], 'IE': [3.0], 'EI': [1.6], 'EE': [1.2]}  
# we = (60 * 0.27 / 10) * mV # excitatory synaptic weight
# wi = (20 * 4.5 / 10) * mV # inhibitory synaptic weight

mean_A_E = 18   # input -> E (ST.input???)
mean_A_I = 9    # input -> I

pars = dict(EE=pars_ee, EI=pars_ei, IE=pars_ie, II=pars_ii)

# transmission delay between liquid
syn_delay_ee = 1.5
syn_delay = .8

def compute_p(coor_neu1, coor_neu2, conn_type='EE'):
    x1, y1, z1 = coor_neu1
    x2, y2, z2 = coor_neu2
    dist = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
    C = pars[conn_type]['C']
    p = C * np.exp(-dist/lbd**2)
    return 1 if p >= np.random.random_sample() else 0

def get_corr():
    pos_corr = []
    for z in range(n_z):
        for y in range(n_y):
            for x in range(n_x):
                pos_corr.append([x, y, z])
    return np.array(pos_corr)

def make_conn(conn_type='EE'):
    neu_coor = get_corr()

    if conn_type[0] == 'I':
        n_pre = num_inh
        coor_pre = neu_coor[i_idx]
        assert n_pre == coor_pre.shape[0]
    elif conn_type[0] == 'E':
        n_pre = num_exc
        coor_pre = neu_coor[e_idx]
        assert n_pre == coor_pre.shape[0]
    
    if conn_type[1] == 'I':
        n_post = num_inh
        coor_post = neu_coor[i_idx]
        assert n_post == coor_post.shape[0]
    elif conn_type[1] == 'E':
        n_post = num_exc
        coor_post = neu_coor[e_idx]
        assert n_post == coor_post.shape[0]

    mat = np.zeros((n_pre, n_post))

    for i in range(n_pre):
        for j in range(n_post):
            mat[i][j] = compute_p(coor_pre[i], coor_post[j], conn_type=conn_type)
    # mat = bp.types.MatConn(mat)
    return mat

def gaussian_sample(mu, num, ratio = .5):
    sigma = mu * ratio
    pars = np.random.normal(mu, sigma, num)
    return np.where(pars>0., pars, np.random.rand(num) * sigma * 2 + (mu-sigma))

def get_weights(A, shape):
    A =  abs(A)
    std = np.random.standard_gamma(A, shape)    # ???
    return np.random.normal(A, std, shape)

# synapse connections
syn = bpmodels.synapses.get_STP()

# E -> E
syn_ee = bp.SynConn(syn,
                    pre_group=liquid_E,
                    post_group=liquid_E, 
                    conn=make_conn('EE'),
                    delay=syn_delay_ee)

n_conn = syn_ee.num
print(n_conn)

syn_ee.pars['U'] = gaussian_sample(mu = pars['EE']['U'], num=n_conn)
syn_ee.pars['tau_d'] = gaussian_sample(mu = pars['EE']['taud'], num=n_conn)
syn_ee.pars['tau_f'] = gaussian_sample(mu = pars['EE']['tauf'], num=n_conn)
syn_ee.ST['w'] = get_weights(pars['EE']['A'], syn_ee.ST['w'].shape)

# E -> I
syn_ei = bp.SynConn(syn,
                    pre_group=liquid_E,
                    post_group=liquid_I, 
                    conn=make_conn('EI'),
                    delay=syn_delay)

n_conn = syn_ei.num
print(n_conn)

syn_ei.pars['U'] = gaussian_sample(mu = pars['EI']['U'], num=n_conn)
syn_ei.pars['tau_d'] = gaussian_sample(mu = pars['EI']['taud'], num=n_conn)
syn_ei.pars['tau_f'] = gaussian_sample(mu = pars['EI']['tauf'], num=n_conn)
syn_ei.ST['w'] = get_weights(pars['EI']['A'], syn_ei.ST['w'].shape)


# I -> E
syn_ie = bp.SynConn(syn,
                    pre_group=liquid_I,
                    post_group=liquid_E, 
                    conn=make_conn('IE'),
                    delay=syn_delay)

n_conn = syn_ie.num
print(n_conn)

syn_ie.pars['U'] = gaussian_sample(mu = pars['IE']['U'], num=n_conn)
syn_ie.pars['tau_d'] = gaussian_sample(mu = pars['IE']['taud'], num=n_conn)
syn_ie.pars['tau_f'] = gaussian_sample(mu = pars['IE']['tauf'], num=n_conn)
syn_ie.ST['w'] = -get_weights(pars['IE']['A'], syn_ie.ST['w'].shape)

# I -> I
syn_ii = bp.SynConn(syn,
                    pre_group=liquid_I,
                    post_group=liquid_I, 
                    conn=make_conn('II'),
                    delay=syn_delay)

n_conn = syn_ii.num
print(n_conn)

syn_ii.pars['U'] = gaussian_sample(mu = pars['II']['U'], num=n_conn)
syn_ii.pars['tau_d'] = gaussian_sample(mu = pars['II']['taud'], num=n_conn)
syn_ii.pars['tau_f'] = gaussian_sample(mu = pars['II']['tauf'], num=n_conn)
syn_ii.ST['w'] = -get_weights(pars['II']['A'], syn_ii.ST['w'].shape)


# run test
I_e = get_weights(mean_A_E, num_exc) + I_b
I_i = get_weights(mean_A_I, num_inh) + I_b

net = bp.Network(liquid_E, liquid_I, syn_ee, syn_ei, syn_ie, syn_ii)
net.run(duration=50., inputs=[(liquid_E, 'ST.input', I_e),
                                (liquid_I, 'ST.input', I_i)], report=True)

# ----------------
# readout synapse
# ----------------
def readout_syn(tau_decay, eta=.001, gamma=.1, mu=1., eps=1e-03, alpha=1., window = 20.):
    @bp.integrate
    def int_s(s, t):
        return - s / tau_decay

    ST = bp.types.SynState('s', w=1, Ib=13.5)

    requires = {
        'pre': bp.types.NeuState('spike'),
        'post': bp.types.NeuState('input', 'spike'),
        'state': bp.types.Array(dim=2),
        'target': bp.types.Float()
    }

    def delta_rule(w, s, rate, target):
        err = rate - target
        z = alpha * np.ones(w.shape[1]+1)
        s = np.append(s, -1)    # append 1 to state for threshold / background current
        s = np.dot(w, s)
        thresh = np.zeros(w.shape[0])
        if abs(err) > eps:
            for idx in range(w.shape[0]):
                delta = None
                if err > eps and s[idx] >= 0. :
                    delta = -z
                elif err < -eps and s[idx] < 0. :
                    delta = z
                elif err <= eps and (s[idx] >=0. and s[idx] < gamma):
                    delta = mu * z
                    M_plus += 1
                elif err >= -eps and (s[idx] < 0. and s[idx] > gamma):
                    delta = mu * -z
                    M_mius += 1
                else:
                    delta = 0.
            wi = w[idx]
            wi = np.append(wi, 1)
            wi2 = np.dot(wi, wi)
            temp = wi - eta * (wi2 - 1.) * wi + eta * delta
            thresh[idx] = temp[-1]
            w[idx] = temp[:-1]
        return w, thresh


    def update(ST, _t, pre, post, state, target):
        s = int_s(ST['s'], _t)
        s += pre['spike']
        ST['s'] = s

        state = np.vstack((state, post['spike']))
        if (state.shape[0]>window):
            state = state[1:]
        state2 = state.sum(axis=0)
        state2 = np.where(state2>0., 1., 0.)
        post_rate = np.mean(state2)
        state2 = np.where(state2>0., 1., -1.)

        ST['w'], ST['Ib'] = delta_rule(ST['w'], state2, post_rate, target)

    @bp.delayed
    def output(ST, post):
        post['input'] += ST['w'] * ST['s'] + ST['Ib']

    return bp.SynType(name='delta_rule',
                      ST=ST, requires=requires,
                      steps=(update, output),
                      mode = "matrix")


# neu_input = bp.inputs.PoissonInput(geometry = num_exc, freqs = 10.)
