# -*- coding: utf-8 -*-

import sys

sys.path.append('/home/annatar/BrainPy/')
sys.path.append('/home/annatar/BrainModels/')

import brainpy as bp


__all__ = [
    'GABAb1',
    'GABAb2',
]

class GABAb1(bp.TwoEndConn):
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
        
        super(GABAb1, self).__init__(pre = pre, post = post, **kwargs)
        
    @staticmethod
    @bp.odeint
    def integral(G, R, t, k1, k2, k3, k4, TT):
        dGdt = k1 * R - k2 * G
        dRdt = k3 * TT * (1 - R) - k4 * R
        return dGdt, dRdt
       
    def update(self, _t):
        spike = bp.backend.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.t_last_pre_spike = bp.backend.where(spike, _t, self.t_last_pre_spike)
        TT = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
        self.G, self.R = self.integral(
            self.G, self.R, _t,
            self.k1, self.k2, 
            self.k3, self.k4, TT)
        self.s = self.G**4 / (self.G**4 + self.kd)
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.backend.sum(self.g.pull(), 0) * (self.post.V - self.E)  


class GABAb2(bp.TwoEndConn):
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

        super(GABAb2, self).__init__(pre = pre, post = post, **kwargs)
        
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

class LIF2(bp.NeuGroup):
    """Leaky Integrate-and-Fire neuron model.
       .. math::
           \\tau \\frac{d V}{d t}=-(V-V_{rest}) + RI(t)
       **Neuron Parameters**
       ============= ============== ======== =========================================
       **Parameter** **Init Value** **Unit** **Explanation**
       ------------- -------------- -------- -----------------------------------------
       V_rest        0.             mV       Resting potential.
       V_reset       -5.            mV       Reset potential after spike.
       V_th          20.            mV       Threshold potential of spike.
       R             1.             \        Membrane resistance.
       tau           10.            \        Membrane time constant. Compute by R * C.
       t_refractory  5.             ms       Refractory period length.(ms)
       noise         0.             \        noise.
       mode          'scalar'       \        Data structure of ST members.
       ============= ============== ======== =========================================
       Returns:
           bp.Neutype: return description of LIF model.
       **Neuron State**
       ST refers to neuron state, members of ST are listed below:
       =============== ================= =========================================================
       **Member name** **Initial Value** **Explanation**
       --------------- ----------------- ---------------------------------------------------------
       V               0.                Membrane potential.
       input           0.                External and synaptic input current.
       spike           0.                Flag to mark whether the neuron is spiking.
                                         Can be seen as bool.
       refractory      0.                Flag to mark whether the neuron is in refractory period.
                                         Can be seen as bool.
       t_last_spike    -1e7              Last spike time stamp.
       =============== ================= =========================================================
       Note that all ST members are saved as floating point type in BrainPy,
       though some of them represent other data types (such as boolean).
       References:
           .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single
                  neurons to networks and models of cognition. Cambridge
                  University Press, 2014.
       """

    target_backend = 'general'

    @staticmethod
    def derivative(V, t, Iext, V_rest, R, tau):
        return (-V + V_rest + R * Iext) / tau

    def __init__(self, size, t_refractory=1., V_rest=0.,
                 V_reset=-5., V_th=20., R=1., tau=10., **kwargs):
        # parameters
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.t_refractory = t_refractory

        # variables
        num = bp.size2len(size)
        self.t_last_spike = bp.backend.ones(num) * -1e7
        self.input = bp.backend.zeros(num)
        self.V = bp.backend.ones(num) * V_reset
        self.refractory = bp.backend.zeros(num, dtype=bool)
        self.spike = bp.backend.zeros(num, dtype=bool)

        self.int_V = bp.odeint(self.derivative)
        super(LIF2, self).__init__(size=size, **kwargs)

    def update(self, _t):
        refractory = (_t - self.t_last_spike) <= self.t_refractory
        V = self.int_V(self.V, _t, self.input, self.V_rest, self.R, self.tau)
        V = bp.backend.where(refractory, self.V, V)
        spike = V >= self.V_th
        self.t_last_spike = bp.backend.where(spike, _t, self.t_last_spike)
        self.V = bp.backend.where(spike, self.V_reset, V)
        self.refractory = refractory
        self.input[:] = 0.
        self.spike = spike

'''
import matplotlib.pyplot as plt

duration = 100.
dt = 0.02
print(bp.__version__)
bp.backend.set('numpy', dt=dt)
size = 10
neu_pre = LIF2(size, monitors = ['V', 'input', 'spike'], )
neu_pre.V_rest = -65.
neu_pre.V_reset = -70.
neu_pre.V_th = -50.
neu_pre.V = bp.backend.ones(size) * -65.
neu_post = LIF(size, monitors = ['V', 'input', 'spike'], )

syn_GABAb = GABAb1(pre = neu_pre, post = neu_post, 
                       conn = bp.connect.One2One(),
                       delay = 10., monitors = ['s'], )


net = bp.Network(neu_pre, syn_GABAb, neu_post)
net.run(200, inputs = [(neu_pre, 'input', 25)], report = True)

print(neu_pre.mon.spike.max())
print(syn_GABAb.mon.s.max())


bp.visualize.line_plot(net.ts, neu_pre.mon.V, show=True)

# paint gabaa
ts = net.ts
#print(gabaa.mon.s.shape)
plt.plot(ts, syn_GABAb.mon.s[:, 0, 0], label='s')
plt.legend()
plt.show()
'''
