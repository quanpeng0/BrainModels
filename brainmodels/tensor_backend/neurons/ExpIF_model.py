# -*- coding: utf-8 -*-

import brainpy as bp


class ExpIF(bp.NeuGroup):
    """Exponential Integrate-and-Fire neuron model.

    .. math::

        \\tau\\frac{d V}{d t}= - (V-V_{rest}) + \\Delta_T e^{\\frac{V-V_T}{\\Delta_T}} + RI(t)

    **Neuron Parameters**

    ============= ============== ======== ===================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ---------------------------------------------------
    V_rest        -65.           mV       Resting potential.

    V_reset       -68.           mV       Reset potential after spike.

    V_th          -30.           mV       Threshold potential of spike.

    V_T           -59.9          mV       Threshold potential of generating action potential.

    delta_T       3.48           \        Spike slope factor.

    R             10.            \        Membrane resistance.

    C             1.             \        Membrane capacitance.

    tau           10.            \        Membrane time constant. Compute by R * C.

    t_refractory  1.7            \        Refractory period length.
    ============= ============== ======== ===================================================

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
        .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation 
               mechanisms determine the neuronal response to fluctuating 
               inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    """
    target_backend = 'general'

    @staticmethod
    def derivative(V, t, I_ext, V_rest, delta_T, V_T, R, tau):
        dvdt = (- V + V_rest \
                + delta_T * bp.backend.exp((V - V_T) / delta_T) + R * I_ext) \
               / tau
        return dvdt

    def __init__(self, size, V_rest=-65., V_reset=-68.,
                 V_th=-30., V_T=-59.9, delta_T=3.48,
                 R=10., C=1., tau=10., t_refractory=1.7,
                 **kwargs):
        # parameters
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_T = V_T
        self.delta_T = delta_T
        self.R = R
        self.C = C
        self.tau = tau
        self.t_refractory = t_refractory

        # variables
        self.V = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)
        self.spike = bp.backend.zeros(size, dtype=bool)
        self.refractory = bp.backend.zeros(size, dtype=bool)
        self.t_last_spike = bp.backend.ones(size) * -1e7

        self.integral = bp.odeint(self.derivative)
        super(ExpIF, self).__init__(size=size, **kwargs)

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
