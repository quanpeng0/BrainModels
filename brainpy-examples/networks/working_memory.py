import brainpy as bp
import numpy as np
import bpmodels
import matplotlib.pyplot as plt
import math

#TODO: check unit!!

# set params
## set global params
dt=0.02
bp.profile.set(jit=True,
               device='cpu',
               dt=dt,
               merge_steps=True)

base_N_E = 2048
base_N_I = 512
scale = 8
N_E = base_N_E//scale
N_I = base_N_I//scale

# ========
#  neuron
# ========

## set neuron params
### E-neurons/pyramidal cells
C_E = 0.5
g_E = 25.
R_E = 1/g_E  #??? check unit here
tau_E = R_E * C_E
V_rest_E = -70.
V_reset_E = -60.
V_th_E = -50.
t_refractory_E = 2.
### I-neurons/interneurons
C_I = 0.2
g_I = 20.
R_I = 1/g_I
tau_I = R_I * C_I
V_rest_I = -70.
V_reset_I = -60.
V_th_I = -50.
t_refractory_I = 1.

def get_LIF(V_rest=0., V_reset=0., V_th=0., R=0., tau=0., t_refractory=0., prefer_cue=0.):
    ST = bp.types.NeuState('V', 'input', 'spike', 'refractory', t_last_spike = -1e7)
    
    @bp.integrate
    def int_V(V, t, I_ext):  # integrate u(t)
        return (- (V - V_rest) + R * I_ext) / tau

    def update(ST, _t):
        # update variables
        ST['spike'] = 0
        if _t - ST['t_last_spike'] <= t_refractory:
            ST['refractory'] = 1.
        else:
            ST['refractory'] = 0.
            V = int_V(ST['V'], _t, ST['input'])
            if V >= V_th:
                V = V_reset
                ST['spike'] = 1
                ST['t_last_spike'] = _t
            ST['V'] = V
        ST['input'] = 0.  # reset input here or it will be brought to next step

    return bp.NeuType(name='LIF_neuron',
                      ST=ST,
                      steps=update,
                      mode='scalar')

LIF = get_LIF()
neu_E = bp.NeuGroup(model = LIF, geometry = N_E, monitors = ['V'])
neu_E.pars['V_rest'] = V_rest_E
neu_E.pars['V_reset'] = V_reset_E
neu_E.pars['V_th'] = V_th_E
neu_E.pars['R'] = R_E
neu_E.pars['tau'] = tau_E
neu_E.pars['t_refractory'] = t_refractory_E
prefer_cue_E = np.random.randint(0, 360, size = N_E)
neu_E.pars['prefer_cue'] = prefer_cue_E  #TODO: assign prefer cue here
neu_I = bp.NeuGroup(model = LIF, geometry = N_I, monitors = ['V'])
neu_I.pars['V_rest'] = V_rest_I
neu_I.pars['V_reset'] = V_reset_I
neu_I.pars['V_th'] = V_th_I
neu_I.pars['R'] = R_I
neu_I.pars['tau'] = tau_I
neu_I.pars['t_refractory'] = t_refractory_I
prefer_cue_I = np.random.randint(0, 360, size = N_I)
neu_I.pars['prefer_cue'] = prefer_cue_I  #TODO: assign prefer cue here

## set input params
poission_frequency = 1800 #or 1000*1.8 #be precise, what is that?
g_max_input2E = 3.1  #AMPA
g_max_input2I = 2.38 #AMPA

neu_input = bp.inputs.PossionInput(geometry = N_E + N_I, freqs = poission_frequency)
syn_input2E = bp.SynConn(model=AMPA, 
                         pre_group = neu_input[:N_E],
                         post_group = neu_E,
                         conn=bp.connect.One2One(), 
                         delay=0.)
#TODO: check how to set delay time?
syn_input2E.pars['tau_decay'] = tau_AMPA
syn_input2E.pars['E'] = E_AMPA
syn_input2E.pars['g_max'] = g_max_input2E
syn_input2I = bp.SynConn(model=AMPA, 
                         pre_group = neu_input[N_E:],
                         post_group = neu_I,
                         conn=bp.connect.One2One(), 
                         delay=0.)
syn_input2I.pars['tau_decay'] = tau_AMPA
syn_input2I.pars['E'] = E_AMPA
syn_input2I.pars['g_max'] = g_max_input2I

## set synapse params
### AMPA
tau_AMPA = 2.
E_AMPA = 0.
### GABAa
tau_GABAa = 10.
E_GABAa = -70.
### NMDA
tau_decay_NMDA = 100.
tau_rise_NMDA = 2.
cc_Mg_NMDA = 1.
alpha_NMDA = 0.062
beta_NMDA = 3.57
a_NMDA = 0.5 #kHz #TODO: check lianggang, 0.5k
E_NMDA = 0.
g_max_E2E = 0.381
g_max_E2I = 0.292
g_max_I2E = 1.336
g_max_I2I = 1.024

# =========
#  synapse
# =========

def get_NMDA(g_max=0., E=E_NMDA, alpha=alpha_NMDA, beta=beta_NMDA, 
             cc_Mg=cc_Mg_NMDA, a=a_NMDA, 
             tau_decay=tau_decay_NMDA, tau_rise=tau_rise_AMDA, 
             mode = 'vector'):

    ST=bp.types.SynState('s', 'x', 'g')

    requires = dict(
        pre=bp.types.NeuState(['spike']),
        post=bp.types.NeuState(['V', 'input'])
    )

    @bp.integrate
    def int_x(x, t):
        return -x / tau_rise

    @bp.integrate
    def int_s(s, t, x):
        return -s / tau_decay + a * x * (1 - s)
    
    if mode == 'scalar':
        def update(ST, _t, pre):
            x = int_x(ST['x'], _t)
            x += pre['spike']
            s = int_s(ST['s'], _t, x)
            ST['x'] = x
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post):
            I_syn = ST['g'] * (post['V'] - E)
            g_inf = 1 / (1 + cc_Mg / beta * np.exp(-alpha * post['V']))
            post['input'] -= I_syn * g_inf

    elif mode == 'vector':
        requires['pre2syn']=bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index')
        requires['post2syn']=bp.types.ListConn(help='Post-synaptic neuron index -> synapse index')

        def update(ST, _t, pre, pre2syn):
            for pre_id in range(len(pre2syn)):
                if pre['spike'][pre_id] > 0.:
                    syn_ids = pre2syn[pre_id]
                    ST['x'][syn_ids] += 1.
            x = int_x(ST['x'], _t)
            s = int_s(ST['s'], _t, x)
            ST['x'] = x
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post, post2syn):
            g = np.zeros(len(post2syn), dtype=np.float_)
            for post_id, syn_ids in enumerate(post2syn):
                g[post_id] = np.sum(ST['g'][syn_ids])    
            I_syn = g * (post['V'] - E)
            g_inf = 1 / (1 + cc_Mg / beta * np.exp(-alpha * post['V']))
            post['input'] -= I_syn * g_inf

    elif mode == 'matrix':
        requires['conn_mat']=bp.types.MatConn()

        def update(ST, _t, pre, conn_mat):
            x = int_x(ST['x'], _t)
            x += pre['spike'].reshape((-1, 1)) * conn_mat
            s = int_s(ST['s'], _t, x)
            ST['x'] = x
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post):
            g = np.sum(ST['g'], axis=0)
            I_syn = g * (post['V'] - E)
            g_inf = 1 / (1 + cc_Mg / beta * np.exp(-alpha * post['V']))
            post['input'] -= I_syn * g_inf

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='NMDA_synapse',
                      ST=ST, requires=requires,
                      steps=(update, output),
                      mode = mode)
                      
syn_E2E = bp.SynConn(model = NMDA, pre_group = neu_E, post_group = neu_E,
                     conn = bp.connect.All2All())
syn_E2E.pars['g_max'] = g_max_E2E * JE2E  # TODO: check? to ignore w, multiply J onto g_max?
syn_E2I = bp.SynConn(model = NMDA, pre_group = neu_E, post_group = neu_I,
                     conn = bp.connect.All2All())
syn_E2I.pars['g_max'] = g_max_E2I * JE2I
syn_I2E = bp.SynConn(model = GABAa, pre_group = neu_I, post_group = neu_E,
                     conn = bp.connect.All2All())
syn_I2E.pars['tau_decay'] = tau_GABAa
syn_I2E.pars['E'] = E_GABAa
syn_I2E.pars['g_max'] = g_max_I2E * JI2E
syn_I2I = bp.SynConn(model = GABAa, pre_group = neu_I, post_group = neu_I,
                     conn = bp.connect.All2All())
syn_I2I.pars['tau_decay'] = tau_GABAa
syn_I2I.pars['E'] = E_GABAa
syn_I2I.pars['g_max'] = g_max_I2I * JI2I

def rotate_distance(x, y):
    dist = np.abs(x - y)
    return min(dist, 360 - dist)

delta_E2E = 18 #TODO: check danwei, 18du vs. 2 pi
J_plus_E2E = 1.62
tmp = math.sqrt(2. * math.pi) * delta_E2E * erf(180. / math.sqrt(2.) / delta_E2E) / 360.
J_neg_E2E = (1. - J_plus_E2E * tmp) / (1. - tmp)
JE2E = []
for i in range(N_E**2):
    JE2E.append(J_neg_E2E + 
                (J_plus_E2E - J_neg_E2E) * 
                np.exp(-rotate_distance(prefer_cue_E[i//N_E], prefer_cue_E[i%N_E])**2/(2 * delta_E2E ** 2)))
JE2E = np.array(JE2E)
#TODO: build connection
JE2I = 1. / (N_E * N_I) #TODO: all set to 1. or normalize?
JI2E = 1. / (N_I * N_E)
JI2I = 1. / (N_I * N_I)

# get neu & syn type
#AMPA = bpmodels.synapses.get_AMPA1()  
GABAa = bpmodels.synapses.get_GABAa()#TODO: mode=?
NMDA = get_NMDA()

# build neuron groups & synapse connections

#TODO: set neuron param here
#TODO: set AMPA param here
#TODO: set GABAa param here
#TODO:set 1800Hz input here
