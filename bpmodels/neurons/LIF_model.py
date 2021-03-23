# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np
from numba import prange
import pdb

bp.backend.set('numba', dt=0.02)

class LIF(bp.NeuGroup):
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
    
    **Neuron Variables**    
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
    
    Both parameters and variables are objects of Class LIF. 
    Note that variables are saved as vectors of NeuGroup size.
        
    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """
    target_backend = 'general'  #TODO: the relation between backend and relization.

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
        self.spike = bp.backend.zeros(size)
        self.refractory = bp.backend.zeros(size)
        self.t_last_spike = bp.backend.ones(size) * -1e7

        super(LIF, self).__init__(size = size, **kwargs)

    @staticmethod
    @bp.odeint()
    def integral(V, t, I_ext, V_rest, R, tau):  # integrate u(t)
        return (- (V - V_rest) + R * I_ext) / tau
    
    def update(self, _t):
        # update variables
        #pdb.set_trace()
        #TODO: may I consider which backend and judge if I need prange?
        for i in prange(self.size[0]):
            spike = 0.
            refractory = (_t - self.t_last_spike[i] <= self.t_refractory)
            if not refractory:
                V = self.integral(self.V[i], _t, self.input[i], self.V_rest, self.R, self.tau)
                spike = (V >= self.V_th)
                if spike:
                    V = self.V_rest
                    self.t_last_spike[i] = _t
                self.V[i] = V
            self.spike[i] = spike
            self.refractory[i] = refractory
            self.input[i] = 0.  # reset input here or it will be brought to next step

if __name__ == "__main__":
    group = LIF(10, monitors=['V'])

    group.run(duration = 200., inputs=('input', 26.), report=True)
    bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

    group.t_refractory = 10.
    group.run(duration = (200., 400.), inputs=('input', 26.), report=True)
    bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)