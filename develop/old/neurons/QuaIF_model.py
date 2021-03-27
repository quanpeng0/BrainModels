# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np
from numba import prange

bp.backend.set('numba', dt = 0.01)

class QuaIF(bp.NeuGroup):
    """Quadratic Integrate-and-Fire neuron model.
        
    .. math::

        \\tau \\frac{d V}{d t}=a_0(V-V_{rest})(V-V_c) + RI(t)
    
    where the parameters are taken to be :math:`a_0` =0.07, and
    :math:`V_c = -50 mV` (Latham et al., 2000 [2]_).
    

    **Neuron Parameters**
    
    ============= ============== ======== ========================================================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
    V_rest        -65.           mV       Resting potential.

    V_reset       -68.           mV       Reset potential after spike.

    V_th          -30.           mV       Threshold potential of spike and reset.

    V_c           -50.           mV       Critical voltage for spike initiation. Must be larger than V_rest.

    a_0           .07            \        Coefficient describes membrane potential update. Larger than 0.

    R             1              \        Membrane resistance.

    tau           10             ms       Membrane time constant. Compute by R * C.

    t_refractory  0              ms       Refractory period length.

    noise         0.             \        the noise fluctuation.

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ========================================================================================================================    
    
    Returns:
        bp.Neutype: return description of QuaIF model.

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
        .. [2]  P. E. Latham, B.J. Richmond, P. Nelson and S. Nirenberg 
                (2000) Intrinsic dynamics in neuronal networks. I. Theory. 
                J. Neurophysiology 83, pp. 808–827. 
    """

    target_backend = 'general'

    def __init__(self, size, V_rest=-65., V_reset=-68., 
                 V_th=-30., V_c=-50.0, a_0 = .07,
                 R=1., tau=10., t_refractory=0., **kwargs):
        
        # parameters
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_c = V_c
        self.a_0 = a_0
        self.R = R
        self.tau = tau
        self.t_refractory = t_refractory

        # variables
        self.V = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)
        self.spike = bp.backend.zeros(size)
        self.refractory = bp.backend.zeros(size)
        self.t_last_spike = bp.backend.ones(size) * -1e7

        super(QuaIF, self).__init__(size = size, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def integral(V, t, I_ext, V_rest, V_c, R, tau, a_0):  # integrate u(t)
        dVdt = (a_0 * (V - V_rest) * (V - V_c) + R * I_ext) / tau
        return dVdt

    def update(self, _t):
        for i in prange(self.size[0]):
            spike = 0.
            refractory = (_t - self.t_last_spike[i] <= self.t_refractory)
            if not refractory:
                V = self.integral(self.V[i], _t, self.input[i], 
                                  self.V_rest, self.V_c, self.R, 
                                  self.tau, self.a_0)
                spike = (V >= self.V_th)
                if spike:
                    V = self.V_rest
                    self.t_last_spike[i] = _t
                self.V[i] = V
            self.spike[i] = spike
            self.refractory[i] = refractory
            self.input[i] = 0.  # reset input here or it will be brought to next step
