# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
import bpmodels
import matplotlib.pyplot as plt
import pdb

# set params
## set global params
dt = 0.02  #ms
method = 'rk4'
bp.profile.set(jit=True,
               device='cpu',
               dt=dt,
               numerical_method=method)

base_N_E = 1600
base_N_I = 400
net_scale = 20
N_E = int(base_N_E // net_scale)
N_I = int(base_N_I // net_scale)

f = 0.15   # proportion of neurons activated by one of the two stimulus
N_A = int(f * N_E)
N_B = int(f * N_E)
N_non = N_E - N_A - N_B
# Note: N_E = N_A + N_B + N_non
print(f"N_E = {N_E} = {N_A} + {N_B} + {N_non}")
print(f"N_I = {N_I}")

time_scale = 10.
pre_period = 1000. / time_scale
stim_period = 1000.
delay_period = 2000. / time_scale
total_period = pre_period + stim_period + delay_period

## set LIF neu params
V_rest_E = -70.   #mV
V_reset_E = -55.  #mV
V_th_E = -50.     #mV
g_E = 25. * 1e-3  #uS
R_E = 1 / g_E     #MOhm
C_E = 0.5         #nF
tau_E = 20.       #ms
t_refractory_E = 2.  #ms
print(f"R_E * C_E = {R_E * C_E} should be equal to tau_E = {tau_E}")

V_rest_I = -70.   #mV
V_reset_I = -55.  #mV
V_th_I = -50.     #mV
g_I = 20. * 1e-3  #uS
R_I = 1 / g_I     #Mohm
C_I = 0.2         #nF
tau_I = 10.       #ms
t_refractory_I = 1. #ms
print(f"R_I * C_I = {R_I * C_I} should be equal to tau_I = {tau_I}")

def get_LIF(V_rest=-70., V_reset=-55., V_th=-50., R=1.,
            tau=10., t_refractory=5., noise=0., mode='scalar'):

    ST = bp.types.NeuState('V', 'input', 'spike', 'refractory', t_last_spike = -1e7)

    @bp.integrate
    def int_V(V, t, I_ext):  # integrate u(t)
        return (- (V - V_rest) + R * I_ext) / tau, noise / tau

    if mode == 'scalar':
        def update(ST, _t):
            # update variables
            if _t - ST['t_last_spike'] <= t_refractory:
                ST['refractory'] = 1.
            else:
                ST['refractory'] = 0.
                V = int_V(ST['V'], _t, ST['input'])
                if V >= V_th:
                    V = V_reset
                    ST['spike'] = 1.
                    ST['t_last_spike'] = _t
                else:
                    ST['spike'] = 0.
                ST['V'] = V
        
        def reset(ST):
            ST['input'] = 0.  # reset input here or it will be brought to next step

    elif mode == 'vector':

        def update(ST, _t):
            V = int_V(ST['V'], _t, ST['input'])
            is_ref = _t - ST['t_last_spike'] < t_refractory
            V = np.where(is_ref, ST['V'], V)
            is_spike = V > V_th
            spike_idx = np.where(is_spike)[0]
            if len(spike_idx):
                V[spike_idx] = V_reset
                is_ref[spike_idx] = 1.
                ST['t_last_spike'][spike_idx] = _t
            ST['V'] = V
            ST['spike'] = is_spike
            ST['refractory'] = is_ref
        
        def reset(ST):
            ST['input'] = 0.  # reset input here or it will be brought to next step
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
    
    return bp.NeuType(name='LIF_neuron',
                      ST=ST,
                      steps=(update, reset),
                      mode=mode)

## set syn params
E_AMPA = 0.   #mV
tau_decay_AMPA = 2.   #ms

E_NMDA = 0.           #mV
alpha_NMDA = 0.062
beta_NMDA = 3.57
cc_Mg_NMDA = 1.       #mM
a_NMDA = 0.5          #kHz/ms^-1
tau_rise_NMDA = 2.    #ms
tau_decay_NMDA = 100. #ms

E_GABAa = -70.        #mV
tau_decay_GABAa = 5.  #ms

delay_syn = 0.5       #ms

## set syn weights (only useful in recurrent E connections!)
w_pos = 1.7
w_neg = 1 - f * (w_pos - 1) / (1 - f)
print(f"the structured weight is: w_pos = {w_pos}, w_neg = {w_neg}")
### inside select group w = w+
### between group / from non-select group to select group w = w-
weight = np.ones((N_E, N_E), dtype = np.float)
for i in range(N_A):
    weight[i, 0 : N_A] = w_pos
    weight[i, N_A : N_A + N_B] = w_neg
for i in range(N_A, N_A+N_B):
    weight[i, N_A : N_A + N_B] = w_pos
    weight[i, 0 : N_A] = w_neg
for i in range(N_A + N_B, N_E):
    weight[i, 0 : N_A + N_B] = w_neg
## TODO: check weight assignment(non2A, non2B)

## set background params
poisson_freq = 2400.  #Hz
g_max_ext2E_AMPA = 2.1  * 1e-3 #* net_scale ##TODO: check if scaled?  #uS
g_max_ext2I_AMPA = 1.62 * 1e-3 #* net_scale #uS

g_max_E2E_AMPA  = 0.05 * 1e-3 * net_scale
g_max_E2E_NMDA  = 0.165* 1e-3 * net_scale
g_max_E2I_AMPA  = 0.04 * 1e-3 * net_scale
g_max_E2I_NMDA  = 0.13 * 1e-3 * net_scale
g_max_I2E_GABAa = 1.3  * 1e-3 * net_scale
g_max_I2I_GABAa = 1.0  * 1e-3 * net_scale
# TODO: check here. the g_maxs are ambiguious.

LIF_neu = get_LIF()
AMPA_syn = bpmodels.synapses.get_AMPA1(mode = 'matrix')
NMDA_syn = bpmodels.synapses.get_NMDA(mode = 'matrix')
GABAa_syn = bpmodels.synapses.get_GABAa1(mode = 'matrix')
## TODO: check mode here!!!

# def neurons
## def E neurons/pyramid neurons
neu_E = bp.NeuGroup(model = LIF_neu, geometry = N_E, monitors = ['spike', 'input'])
neu_E.set_schedule(['input', 'update', 'monitor', 'reset'])
neu_E.pars['V_rest'] = V_rest_E
neu_E.pars['V_reset'] = V_reset_E
neu_E.pars['V_th'] = V_th_E
neu_E.pars['R'] = R_E
neu_E.pars['tau'] = tau_E
neu_E.pars['t_refractory'] = t_refractory_E
neu_E.ST['V'] = V_rest_E

## def I neurons/interneurons
neu_I = bp.NeuGroup(model = LIF_neu, geometry = N_I, monitors = ['V'])
neu_I.set_schedule(['input', 'update', 'monitor', 'reset'])
neu_I.pars['V_rest'] = V_rest_I
neu_I.pars['V_reset'] = V_reset_I
neu_I.pars['V_th'] = V_th_I
neu_I.pars['R'] = R_I
neu_I.pars['tau'] = tau_I
neu_I.pars['t_refractory'] = t_refractory_I
neu_I.ST['V'] = V_rest_I

# def synapse connections
## define syn conns between neu_E and neu_I
syn_E2E_AMPA = bp.SynConn(model = AMPA_syn, 
                          pre_group = neu_E, post_group = neu_E, 
                          conn = bp.connect.All2All(),
                          delay = 0.5)
syn_E2E_AMPA.pars['g_max'] = g_max_E2E_AMPA * weight
syn_E2E_AMPA.pars['E'] = E_AMPA
syn_E2E_AMPA.pars['tau_decay'] = tau_decay_AMPA

syn_E2E_NMDA = bp.SynConn(model = NMDA_syn, 
                          pre_group = neu_E, post_group = neu_E, 
                          conn = bp.connect.All2All(),
                          delay = 0.5)
syn_E2E_NMDA.pars['g_max'] = g_max_E2E_NMDA * weight
syn_E2E_NMDA.pars['E'] = E_NMDA
syn_E2E_NMDA.pars['alpha'] = alpha_NMDA
syn_E2E_NMDA.pars['beta'] = beta_NMDA
syn_E2E_NMDA.pars['cc_Mg'] = cc_Mg_NMDA
syn_E2E_NMDA.pars['tau_decay'] = tau_decay_NMDA
syn_E2E_NMDA.pars['a'] = a_NMDA
syn_E2E_NMDA.pars['tau_rise'] = tau_rise_NMDA

syn_E2I_AMPA = bp.SynConn(model = AMPA_syn, 
                          pre_group = neu_E, post_group = neu_I, 
                          conn = bp.connect.All2All(),
                          delay = 0.5)
syn_E2I_AMPA.pars['g_max'] = g_max_E2I_AMPA
syn_E2I_AMPA.pars['E'] = E_AMPA
syn_E2I_AMPA.pars['tau_decay'] = tau_decay_AMPA

syn_E2I_NMDA = bp.SynConn(model = NMDA_syn, 
                          pre_group = neu_E, post_group = neu_I, 
                          conn = bp.connect.All2All(),
                          delay = 0.5)
syn_E2I_NMDA.pars['g_max'] = g_max_E2I_NMDA
syn_E2I_NMDA.pars['E'] = E_NMDA
syn_E2I_NMDA.pars['alpha'] = alpha_NMDA
syn_E2I_NMDA.pars['beta'] = beta_NMDA
syn_E2I_NMDA.pars['cc_Mg'] = cc_Mg_NMDA
syn_E2I_NMDA.pars['tau_decay'] = tau_decay_NMDA
syn_E2I_NMDA.pars['a'] = a_NMDA
syn_E2I_NMDA.pars['tau_rise'] = tau_rise_NMDA

syn_I2E_GABAa = bp.SynConn(model = GABAa_syn,
                           pre_group = neu_I, post_group = neu_E,
                           conn = bp.connect.All2All(),
                           delay = 0.5)
syn_I2E_GABAa.pars['g_max'] = g_max_I2E_GABAa
syn_I2E_GABAa.pars['E'] = E_GABAa
syn_I2E_GABAa.pars['tau_decay'] = tau_decay_GABAa

syn_I2I_GABAa = bp.SynConn(model = GABAa_syn,
                           pre_group = neu_I, post_group = neu_I,
                           conn = bp.connect.All2All(),
                           delay = 0.5)
syn_I2I_GABAa.pars['g_max'] = g_max_I2I_GABAa
syn_I2I_GABAa.pars['E'] = E_GABAa
syn_I2I_GABAa.pars['tau_decay'] = tau_decay_GABAa

## def poisson input
neu_poisson = bp.inputs.PoissonInput(geometry = N_E + N_I, freqs = poisson_freq)
syn_back2E_AMPA = bp.SynConn(model = AMPA_syn,
                              pre_group = neu_poisson[:N_E],
                              post_group = neu_E,
                              conn = bp.connect.One2One())  ##TODO: how to set delay here?
syn_back2E_AMPA.pars['g_max'] = g_max_ext2E_AMPA
syn_back2E_AMPA.pars['E'] = E_AMPA
syn_back2E_AMPA.pars['tau_decay'] = tau_decay_AMPA

syn_back2I_AMPA = bp.SynConn(model = AMPA_syn,
                              pre_group = neu_poisson[N_E:],
                              post_group = neu_I,
                              conn = bp.connect.One2One())  ##TODO: how to set delay here?
syn_back2I_AMPA.pars['g_max'] = g_max_ext2I_AMPA
syn_back2I_AMPA.pars['E'] = E_AMPA
syn_back2I_AMPA.pars['tau_decay'] = tau_decay_AMPA
## TODO: check which neurons receive 2400Hz background possion inputs? N_E? N_E+N_I? 2fN_E +N_I?

## def stimulus input
# Note: inputs only given to A and B group
mu_0 = 40.
coherence = 12.8  #TODO: 0.xx or xx%
rou_A = mu_0/100.
rou_B = mu_0/100.
mu_A = mu_0 + rou_A * coherence
mu_B = mu_0 - rou_B * coherence
print(f"coherence = {coherence}, mu_A = {mu_A}, mu_B = {mu_B}")

def get_poisson(t_start = 0., t_end = 0., freq = 0., mode='vector'):

    ST = bp.types.NeuState('spike')

    # neuron model
    dt = bp.profile.get_dt() / 1000.

    def update(ST, _t):
        if _t > t_start and _t < t_end:
            ST['spike'] = np.random.random(ST['spike'].shape) < freq * dt
        else:
            ST['spike'] = 0.

    return bp.NeuType(name='poisson_input',
                      ST=ST,
                      steps=update,
                      mode=mode)

possion_neu_2A = get_poisson(t_start = pre_period, 
                             t_end = pre_period + stim_period, 
                             freq = mu_A)
possion_neu_2B = get_poisson(t_start = pre_period, 
                             t_end = pre_period + stim_period, 
                             freq = mu_B)
neu_input2A = bp.NeuGroup(model = possion_neu_2A, geometry = N_A)
neu_input2B = bp.NeuGroup(model = possion_neu_2B, geometry = N_B)

syn_input2A_AMPA = bp.SynConn(model = AMPA_syn, 
                         pre_group = neu_input2A,
                         post_group = neu_E[0:N_A],
                         conn = bp.connect.One2One())
syn_input2A_AMPA.pars['g_max'] = g_max_ext2E_AMPA
syn_input2A_AMPA.pars['E'] = E_AMPA
syn_input2A_AMPA.pars['tau_decay'] = tau_decay_AMPA

syn_input2B_AMPA = bp.SynConn(model = AMPA_syn, 
                         pre_group = neu_input2B,
                         post_group = neu_E[N_A:N_A+N_B],
                         conn = bp.connect.One2One())
syn_input2B_AMPA.pars['g_max'] = g_max_ext2E_AMPA
syn_input2B_AMPA.pars['E'] = E_AMPA
syn_input2B_AMPA.pars['tau_decay'] = tau_decay_AMPA

net = bp.Network(neu_poisson, 
                 syn_back2E_AMPA, syn_back2I_AMPA,
                 neu_input2A, neu_input2B, 
                 syn_input2A_AMPA, syn_input2B_AMPA,
                 neu_E, neu_I,
                 syn_E2E_AMPA, syn_E2E_NMDA, 
                 syn_E2I_AMPA, syn_E2I_NMDA,
                 syn_I2E_GABAa, syn_I2I_GABAa)

net.run(duration = total_period, inputs = [], report = True, report_percent = 0.01)

pdb.set_trace()
# visualize
fig, gs = bp.visualize.get_figure(3, 1, 4, 10)

fig.add_subplot(gs[0, 0])
#bp.visualize.raster_plot(net.ts, neu_E.mon.spike[:, 0 : N_A], markersize=1)  #???
bp.visualize.raster_plot(net.ts, neu_E.mon.spike, markersize=1)  #???

fig.add_subplot(gs[1, 0])
plt.plot(net.ts, neu_E.mon.input[:, N_A - 3])
fig.add_subplot(gs[2, 0])
plt.plot(net.ts, neu_E.mon.input[:, N_A + 3])

plt.show()
'''
fig.add_subplot(gs[0, 0])
bp.visualize.raster_plot(net.ts, neu_E[N_A:N_A + N_B].mon.spike, markersize=1)
plt.show()'''