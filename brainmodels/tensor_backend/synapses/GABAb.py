# -*- coding: utf-8 -*-

import brainpy as bp
from numba import prange

__all__ = [
    'GABAb1_vec',
    'GABAb1_mat',
    'GABAb2_vec',
    'GABAb2_mat',
]

class GABAb1_vec(bp.TwoEndConn):
    """GABAb conductance-based synapse model(type 1).

    .. math::

        &\\frac{d[R]}{dt} = k_3 [T](1-[R])- k_4 [R]

        &\\frac{d[G]}{dt} = k_1 [R]- k_2 [G]

        I_{GABA_{B}} &=\\overline{g}_{GABA_{B}} (\\frac{[G]^{4}} {[G]^{4}+K_{d}}) (V-E_{GABA_{B}})


    - [G] is the concentration of activated G protein.
    - [R] is the fraction of activated receptor.
    - [T] is the transmitter concentration.

    **Synapse Parameters**

    ============= ============== ======== ============================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ----------------------------------------------------------------------------
    g_max         0.02           \        Maximum synapse conductance.

    E             -95.           mV       Reversal potential of synapse.

    k1            0.18           \        Activating rate constant of G protein catalyzed 

                                          by activated GABAb receptor.

    k2            0.034          \        De-activating rate constant of G protein.

    k3            0.09           \        Activating rate constant of GABAb receptor.

    k4            0.0012         \        De-activating rate constant of GABAb receptor.

    kd            100.           \        Dissociation rate constant of the binding of 

                                          G protein on K+ channels.

    T             0.5            \        Transmitter concentration when synapse is 

                                          triggered by a pre-synaptic spike.

    T_duration    0.3            \        Transmitter concentration duration time 

                                          after being triggered.
    ============= ============== ======== ============================================================================

    **Synapse Variables**    

    An object of synapse class record those variables for each synapse:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    R                  0.                The fraction of activated receptor.

    G                  0.                The concentration of activated G protein.

    g                  0.                Synapse conductance on post-synaptic neuron.

    t_last_pre_spike   -1e7              Last spike time stamp of pre-synaptic neuron.
    ================== ================= =========================================================

    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """
    target_backend = 'general'

    def __init__(self, pre, post, conn, delay = 0., 
                 g_max=0.02, E=-95., 
                 k1=0.18, k2=0.034, k3=0.09, k4=0.0012,
                 kd=100., T=0.5, T_duration=0.3, **kwargs):
        #params
        self.g_max = g_max
        self.E = E
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.kd = kd
        self.T = T
        self.T_duration = T_duration

        #conns
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        #data
        self.R = bp.backend.zeros(self.size)
        self.G = bp.backend.zeros(self.size)
        self.t_last_pre_spike = bp.backend.ones(self.size) * -1e7
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(GABAb1_vec, self).__init__(pre = pre, post = post, **kwargs)

    @staticmethod
    @bp.odeint()
    def integral(R, G, t, k3, TT, k4, k1, k2):
        dRdt = k3 * TT * (1 - R) - k4 * R
        dGdt = k1 * R - k2 * G
        return dRdt, dGdt

    def update(self, _t):
        #pdb.set_trace()
        for i in prange(self.size): #i is the No. of syn
            pre_id = self.pre_ids[i]  #pre_id is the No. of pre neu
            if self.pre.spike[pre_id]:
                self.t_last_pre_spike[i] = _t
            TT = ((_t - self.t_last_pre_spike[i]) < self.T_duration) * self.T
            R, G = self.integral(self.R[i], self.G[i], _t,
                                 self.k3, TT, self.k4, 
                                 self.k1, self.k2)
            self.R[i] = R
            self.G[i] = G
            self.s[i] = G ** 4 / (G ** 4 + self.kd)
            self.g.push(i, self.g_max * self.s[i])
            post_id = self.post_ids[i]
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)

class GABAb1_mat(bp.TwoEndConn):
    """GABAb conductance-based synapse model(type 1).

    .. math::

        &\\frac{d[R]}{dt} = k_3 [T](1-[R])- k_4 [R]

        &\\frac{d[G]}{dt} = k_1 [R]- k_2 [G]

        I_{GABA_{B}} &=\\overline{g}_{GABA_{B}} (\\frac{[G]^{4}} {[G]^{4}+K_{d}}) (V-E_{GABA_{B}})


    - [G] is the concentration of activated G protein.
    - [R] is the fraction of activated receptor.
    - [T] is the transmitter concentration.

    **Synapse Parameters**

    ============= ============== ======== ============================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ----------------------------------------------------------------------------
    g_max         0.02           \        Maximum synapse conductance.

    E             -95.           mV       Reversal potential of synapse.

    k1            0.18           \        Activating rate constant of G protein catalyzed 

                                          by activated GABAb receptor.

    k2            0.034          \        De-activating rate constant of G protein.

    k3            0.09           \        Activating rate constant of GABAb receptor.

    k4            0.0012         \        De-activating rate constant of GABAb receptor.

    kd            100.           \        Dissociation rate constant of the binding of 

                                          G protein on K+ channels.

    T             0.5            \        Transmitter concentration when synapse is 

                                          triggered by a pre-synaptic spike.

    T_duration    0.3            \        Transmitter concentration duration time 

                                          after being triggered.
    ============= ============== ======== ============================================================================

    **Synapse Variables**    

    An object of synapse class record those variables for each synapse:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    R                  0.                The fraction of activated receptor.

    G                  0.                The concentration of activated G protein.

    g                  0.                Synapse conductance on post-synaptic neuron.

    t_last_pre_spike   -1e7              Last spike time stamp of pre-synaptic neuron.
    ================== ================= =========================================================

    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """
    target_backend = 'general'
    
    def __init__(self, pre, post, conn, delay = 0.,
                 g_max=0.02, E=-95., k1=0.18, k2=0.034, 
	             k3=0.09, k4=0.0012, kd=100., 
	             T=0.5, T_duration=0.3, **kwargs):
        self.g_max = g_max
        self.E = E
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.kd = kd
        self.T = T
        self.T_duration = T_duration
        self.delay = delay
        
        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)
        
        self.R = bp.backend.zeros(self.size)
        self.G = bp.backend.zeros(self.size)
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size = self.size, delay_time = delay)
        self.t_last_pre_spike = bp.backend.ones(self.size) * -1e7
        
        super(GABAb1_mat, self).__init__(pre = pre, post = post, **kwargs)
        
    @staticmethod
    @bp.odeint
    def integral(G, R, t, k1, k2, k3, k4, TT):
        dGdt = k1 * R - k2 * G
        dRdt = k3 * TT * (1 - R) - k4 * R
        return dGdt, dRdt
       
    def update(self, _t):
        spike = bp.backend.reshape(self.pre.spike, (-1, 1)) * self.conn_mat
        self.t_last_pre_spike = bp.backend.where(spike, _t, self.t_last_pre_spike)
        TT = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
        self.G, self.R = self.integral(
            self.G, self.R, _t,
            self.k1, self.k2, 
            self.k2, self.k4, TT)
        self.s = self.G**4 / (self.G**4 + self.kd)
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.backend.sum(self.g.pull(), 0) * (self.post.V - self.E)  


class GABAb2_vec(bp.TwoEndConn):
    """
    GABAb conductance-based synapse model (markov form).

    G-protein cascade occurs in the following steps: 
    (i) the transmitter binds to the receptor, leading to its activated form; 
    (ii) the activated receptor catalyzes the activation of G proteins; 
    (iii) G proteins bind to open K+ channel, with n(=4) independent binding sites.

    .. math::

        &\\frac{d[D]}{dt}=K_{4}[R]-K_{3}[D]

        &\\frac{d[R]}{dt}=K_{1}[T](1-[R]-[D])-K_{2}[R]+K_{3}[D]

        &\\frac{d[G]}{dt}=K_{5}[R]-K_{6}[G]

        I_{GABA_{B}}&=\\bar{g}_{GABA_{B}} \\frac{[G]^{n}}{[G]^{n}+K_{d}}(V-E_{GABA_{B}})

    - [R] is the fraction of activated receptor.
    - [D] is the fraction of desensitized receptor.
    - [G] is the concentration of activated G-protein (μM).
    - [T] is the transmitter concentration.

    **Synapse Parameters**

    ============= ============== ======== ============================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ----------------------------------------------------------------------------
    g_max         0.02           \        Maximum synapse conductance.

    E             -95.           mV       Reversal potential of synapse.

    k1            0.66           \        Activating rate constant of G protein 

                                          catalyzed by activated GABAb receptor.

    k2            0.02           \        De-activating rate constant of G protein.

    k3            0.0053         \        Activating rate constant of GABAb receptor.

    k4            0.017          \        De-activating rate constant of GABAb receptor.

    k5            8.3e-5         \        Activating rate constant of G protein 

                                          catalyzed by activated GABAb receptor.

    k6            7.9e-3         \        De-activating rate constant of activated G protein.

    kd            100.           \        Dissociation rate constant of the binding of 

                                          G protein on K+ channels.

    T             0.5            \        Transmitter concentration when synapse 

                                          is triggered by a pre-synaptic spike.

    T_duration    0.5            \        Transmitter concentration duration time 

                                          after being triggered.
    ============= ============== ======== ============================================================================

    **Synapse Variables**    

    An object of synapse class record those variables for each synapse:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    D                0.                The fraction of desensitized receptor.

    R                0.                The fraction of activated receptor.

    G                0.                The concentration of activated G protein.

    g                0.                Synapse conductance on post-synaptic neuron.

    t_last_pre_spike -1e7              Last spike time stamp of pre-synaptic neuron.
    ================ ================= =========================================================

    References:
        .. [1] Destexhe, Alain, et al. "G-protein activation kinetics and 
               spillover of GABA may account for differences between 
               inhibitory responses in the hippocampus and thalamus." 
               Proc. Natl. Acad. Sci. USA v92 (1995): 9515-9519.

    """
    target_backend = 'general'
    
    def __init__(self, pre, post, conn, delay = 0.,
                 g_max=0.02, E=-95., k1=0.66, k2=0.02, 
                 k3=0.0053, k4=0.017, k5=8.3e-5, k6=7.9e-3, 
                 kd=100., T=0.5, T_duration=0.5,
                 **kwargs):
        #params
        self.g_max = g_max
        self.E = E
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.kd = kd
        self.T = T
        self.T_duration = T_duration

        #conns
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        #vars
        self.D = bp.backend.zeros(self.size)
        self.R = bp.backend.zeros(self.size)
        self.G = bp.backend.zeros(self.size)
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time = delay)
        self.t_last_pre_spike = bp.backend.ones(self.size) * -1e7

        super(GABAb2_vec, self).__init__(pre = pre, post = post, **kwargs)
    
    @staticmethod
    @bp.odeint
    def integral(R, D, G, t, k1, k2, k3, TT, k4, k5, k6):
        dRdt = k1 * TT * (1 - R - D) - k2 * R + k3 * D
        dDdt = k4 * R - k3 * D
        dGdt = k5 * R - k6 * G
        return dRdt, dDdt, dGdt

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            post_id = self.post_ids[i]
            if self.pre.spike[pre_id]:
                self.t_last_pre_spike[i] = _t
            T = ((_t - self.t_last_pre_spike[i]) < self.T_duration) * self.T
            self.R[i], self.D[i], G = self.integral(
                self.R[i], self.D[i], self.G[i], _t,
                self.k1, self.k2, self.k3, T, self.k4, self.k5, self.k6
            )
            self.s[i] = (G ** 4 / (G ** 4 + self.kd))
            self.G[i] = G
            self.g.push(i, self.g_max * self.s[i])
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)


class GABAb2_mat(bp.TwoEndConn):
    """
    GABAb conductance-based synapse model (markov form).

    G-protein cascade occurs in the following steps: 
    (i) the transmitter binds to the receptor, leading to its activated form; 
    (ii) the activated receptor catalyzes the activation of G proteins; 
    (iii) G proteins bind to open K+ channel, with n(=4) independent binding sites.

    .. math::

        &\\frac{d[D]}{dt}=K_{4}[R]-K_{3}[D]

        &\\frac{d[R]}{dt}=K_{1}[T](1-[R]-[D])-K_{2}[R]+K_{3}[D]

        &\\frac{d[G]}{dt}=K_{5}[R]-K_{6}[G]

        I_{GABA_{B}}&=\\bar{g}_{GABA_{B}} \\frac{[G]^{n}}{[G]^{n}+K_{d}}(V-E_{GABA_{B}})

    - [R] is the fraction of activated receptor.
    - [D] is the fraction of desensitized receptor.
    - [G] is the concentration of activated G-protein (μM).
    - [T] is the transmitter concentration.

    **Synapse Parameters**

    ============= ============== ======== ============================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ----------------------------------------------------------------------------
    g_max         0.02           \        Maximum synapse conductance.

    E             -95.           mV       Reversal potential of synapse.

    k1            0.66           \        Activating rate constant of G protein 

                                          catalyzed by activated GABAb receptor.

    k2            0.02           \        De-activating rate constant of G protein.

    k3            0.0053         \        Activating rate constant of GABAb receptor.

    k4            0.017          \        De-activating rate constant of GABAb receptor.

    k5            8.3e-5         \        Activating rate constant of G protein 

                                          catalyzed by activated GABAb receptor.

    k6            7.9e-3         \        De-activating rate constant of activated G protein.

    kd            100.           \        Dissociation rate constant of the binding of 

                                          G protein on K+ channels.

    T             0.5            \        Transmitter concentration when synapse 

                                          is triggered by a pre-synaptic spike.

    T_duration    0.5            \        Transmitter concentration duration time 

                                          after being triggered.
    ============= ============== ======== ============================================================================

    **Synapse Variables**    

    An object of synapse class record those variables for each synapse:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    D                0.                The fraction of desensitized receptor.

    R                0.                The fraction of activated receptor.

    G                0.                The concentration of activated G protein.

    g                0.                Synapse conductance on post-synaptic neuron.

    t_last_pre_spike -1e7              Last spike time stamp of pre-synaptic neuron.
    ================ ================= =========================================================

    References:
        .. [1] Destexhe, Alain, et al. "G-protein activation kinetics and 
               spillover of GABA may account for differences between 
               inhibitory responses in the hippocampus and thalamus." 
               Proc. Natl. Acad. Sci. USA v92 (1995): 9515-9519.

    """
    target_backend = 'general'
    
    def __init__(self, pre, post, conn, delay = 0.,
                 g_max=0.02, E=-95., k1=0.66, k2=0.02, 
                 k3=0.0053, k4=0.017, k5=8.3e-5, k6=7.9e-3, 
                 kd=100., T=0.5, T_duration=0.5,
                 **kwargs):
        #params
        self.g_max = g_max
        self.E = E
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.kd = kd
        self.T = T
        self.T_duration = T_duration

        #conns
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        #vars
        self.D = bp.backend.zeros(self.size)
        self.R = bp.backend.zeros(self.size)
        self.G = bp.backend.zeros(self.size)
        self.s = bp.backend.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time = delay)
        self.t_last_pre_spike = bp.backend.ones(self.size) * -1e7

        super(GABAb2_mat, self).__init__(pre = pre, post = post, **kwargs)
        
    @staticmethod
    @bp.odeint
    def integral(R, D, G, t, k1, k2, k3, TT, k4, k5, k6):
        dRdt = k1 * TT * (1 - R - D) - k2 * R + k3 * D
        dDdt = k4 * R - k3 * D
        dGdt = k5 * R - k6 * G
        return dRdt, dDdt, dGdt

    def update(self, _t):
        spike = bp.backend.reshape(self.pre.spike, (-1, 1)) * self.conn_mat
        self.t_last_pre_spike = bp.backend.where(spike, _t, self.t_last_pre_spike)
        T = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
        self.R, self.D, self.G = self.integral(
            self.R, self.D, self.G, _t,
            self.k1, self.k2, self.k3, T,
            self.k4, self.k5, self.k6)
        self.s = (self.G ** 4 / (self.G ** 4 + self.kd))
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.backend.sum(self.g.pull(), axis=0) * (self.post.V - self.E)
        
'''
class LIF(bp.NeuGroup):
    target_backend = 'general'

    def __init__(self, size, V_rest = 0., V_reset= -5., 
                 V_th = 20., R = 1., tau = 10., 
                 t_refractory = 5., **kwargs):
        
        # parameters
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.t_refractory = t_refractory

        # variables
        self.V = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)
        self.spike = bp.backend.zeros(size, dtype = bool)
        self.refractory = bp.backend.zeros(size, dtype = bool)
        self.t_last_spike = bp.backend.ones(size) * -1e7

        super(LIF, self).__init__(size = size, **kwargs)

    @staticmethod
    @bp.odeint()
    def integral(V, t, I_ext, V_rest, R, tau): 
        return (- (V - V_rest) + R * I_ext) / tau
    
    def update(self, _t):
        # update variables
        not_ref = (_t - self.t_last_spike > self.t_refractory)
        self.V[not_ref] = self.integral(
            self.V[not_ref], _t, self.input[not_ref],
            self.V_rest, self.R, self.tau)
        sp = (self.V > self.V_th)
        self.V[sp] = self.V_reset
        self.t_last_spike[sp] = _t
        self.spike = sp
        self.refractory = ~not_ref
        self.input[:] = 0.

import brainpy as bp
import matplotlib.pyplot as plt

duration = 100.
dt = 0.02
print(bp.__version__)
bp.backend.set('numpy', dt=dt)
size = 10
neu_pre = LIF(size, monitors = ['V', 'input', 'spike'], show_code = True)
neu_pre.V_rest = -65.
neu_pre.V_reset = -70.
neu_pre.V_th = -50.
neu_pre.V = bp.backend.ones(size) * -65.
neu_post = LIF(size, monitors = ['V', 'input', 'spike'], show_code = True)

syn_GABAb = GABAb1_vec(pre = neu_pre, post = neu_post, 
                       conn = bp.connect.One2One(),
                       delay = 10., monitors = ['s'], show_code = True)

current, dur = bp.inputs.constant_current([(21., 20.), (0., duration - 20.)])
net = bp.Network(neu_pre, syn_GABAb, neu_post)
net.run(dur, inputs = [(neu_pre, 'input', current)], report = True)

# paint gabaa
ts = net.ts
fig, gs = bp.visualize.get_figure(2, 1, 5, 6)

#print(gabaa.mon.s.shape)
fig.add_subplot(gs[0, 0])
plt.plot(ts, syn_GABAb.mon.s[:, 0], label='s')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(ts, neu_post.mon.V[:, 0], label='post.V')
plt.legend()

plt.show()'''
