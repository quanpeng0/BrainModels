# -*- coding: utf-8 -*-

import brainpy as bp


class LIF(bp.NeuGroup):
    """
    Leaky Integrate-and-Fire neuron model.
   
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
    ============= ============== ======== =========================================
    
    **Neuron Variables**    

    An object of neuron class record those variables for each neuron:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                  0.                Membrane potential.
       
    input              0.                External and synaptic input current.
    
    spike              0.                Flag to mark whether the neuron is spiking.

                                         Can be seen as bool.
    
    refractory         0.                Flag to mark whether the neuron is in refractory period.
     
                                         Can be seen as bool.

    t_last_spike       -1e7              Last spike time stamp.
    ================== ================= =========================================================
    
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

        self.integral = bp.odeint(self.derivative)
        super(LIF, self).__init__(size=size, **kwargs)

    def update(self, _t):
        refractory = (_t - self.t_last_spike) <= self.t_refractory
        V = self.integral(self.V, _t, self.input, self.V_rest, self.R, self.tau)
        V = bp.backend.where(refractory, self.V, V)
        spike = self.V_th <= V
        self.t_last_spike = bp.backend.where(spike, _t, self.t_last_spike)
        self.V = bp.backend.where(spike, self.V_reset, V)
        self.refractory = refractory
        self.input[:] = 0.
        self.spike = spike
