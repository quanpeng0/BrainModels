# -*- coding: utf-8 -*-

import brainpy as bp
from numba import prange

__all__ = [
    'AdQuaIF'
]

class AdQuaIF(bp.NeuGroup):
    """Adaptive Quadratic Integrate-and-Fire neuron model.
        
    .. math::

        \\tau_m \\frac{d V}{d t}=a_0(V-V_{rest})(V-V_c) - R w + RI(t)
        
        \\tau_w \\frac{d w}{d t}=a(V-V_{rest}) - w + b \\tau_w \\sum \\delta (t-t^f)


    **Neuron Parameters**
    
    ============= ============== ======== ========================================================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
    V_rest        -65.           mV       Resting potential.

    V_reset       -68.           mV       Reset potential after spike.

    V_th          -30.           mV       Threshold potential of spike and reset.

    a_0           .07            \        Coefficient describes membrane potential update. Larger than 0.

    V_c           -50.           mV       Critical voltage for spike initiation. Must be larger than V_rest.

    a             1              \        The sensitivity of the recovery variable :math:`u` to the sub-threshold fluctuations of the membrane potential :math:`v`

    b             .1             \        The increment of :math:`w` produced by a spike.

    R             1              \        Membrane resistance.

    tau           10             ms       Membrane time constant. Compute by R * C.

    tau_w         10             ms       Time constant of the adaptation current.

    t_refractory  0              ms       Refractory period length.

    noise         0.             \        the noise fluctuation.

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ========================================================================================================================    
    
    Returns:
        bp.Neutype: return description of the AdQuaIF model.

    **Neuron State**

    ST refers to neuron state, members of ST are listed below:
    
    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    V               0.                Membrane potential.

    w               0.                Adaptation current.
    
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

    def __init__(self, size, V_rest=-65., V_reset=-68., 
                 V_th=-30., V_c=-50.0, a_0 = .07,
                 a = 1., b=.1, R=1., tau=10., tau_w = 10.,
                 t_refractory=0., **kwargs):
        
        # parameters
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_c = V_c
        self.a_0 = a_0
        self.a = a
        self.b = b
        self.R = R
        self.tau = tau
        self.tau_w = tau_w
        self.t_refractory = t_refractory

        # variables
        num = bp.size2len(size)
        self.V = bp.backend.ones(num) * V_reset
        self.w = bp.backend.zeros(size)
        self.input = bp.backend.zeros(num)
        self.spike = bp.backend.zeros(num, dtype=bool)
        self.refractory = bp.backend.zeros(num, dtype=bool)
        self.t_last_spike = bp.backend.ones(num) * -1e7

        super(AdQuaIF, self).__init__(size = size, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def integral(V, w, t, I_ext, V_rest, V_c, R, tau, tau_w, a, a_0):  # integrate u(t)
        dwdt = (a * (V - V_rest) - w) / tau_w
        dVdt = (a_0 * (V - V_rest) * (V - V_c) - R * w + R * I_ext) / tau
        return dVdt, dwdt

    def update(self, _t):
        for i in prange(self.size[0]):
            spike = 0.
            refractory = (_t - self.t_last_spike[i] <= self.t_refractory)
            if not refractory:
                V, w = self.integral(self.V[i], self.w[i], _t, self.input[i], 
                                  self.V_rest, self.V_c, self.R, 
                                  self.tau, self.tau_w, self.a, self.a_0)
                spike = (V >= self.V_th)
                if spike:
                    V = self.V_rest
                    w += self.b
                    self.t_last_spike[i] = _t
                self.V[i] = V
                self.w[i] = w
            self.spike[i] = spike
            self.refractory[i] = refractory
            self.input[i] = 0.  # reset input here or it will be brought to next step
