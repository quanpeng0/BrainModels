# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np


def get_LIF(V_rest=0., V_reset=-5., V_th=20., R=1.,
            tau=10., t_refractory=5., noise=0., mode='scalar'):
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

    R             1.             /        Membrane resistance.

    tau           10.            /        Membrane time constant. Compute by R * C.

    t_refractory  5.             ms       Refractory period length.(ms)

    noise         0.             /        noise.

    mode          'scalar'       /        Data structure of ST members.
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

    ST = bp.types.NeuState('V', 'input', 'spike', 'refractory', t_last_spike=-1e7)

    @bp.integrate
    def int_V(V, t, I_ext):  # integrate u(t)
        return (- (V - V_rest) + R * I_ext) / tau, noise / tau

    if mode == 'scalar':
        def update(ST, _t):
            # update variables
            if _t - ST['t_last_spike'] < t_refractory:
                ST['refractory'] = 1.
            else:
                ST['refractory'] = 0.
                V = int_V(ST['V'], _t, ST['input'])
                if V >= V_th:
                    V = V_reset
                    ST['spike'] = 1
                    ST['t_last_spike'] = _t
                else:
                    ST['spike'] = 0.
                ST['V'] = V
            ST['input'] = 0.  # reset input here or it will be brought to next step

        return bp.NeuType(name='LIF_neuron',
                          ST=ST,
                          steps=update,
                          mode=mode)

    elif mode == 'vector':

        def update(ST, _t):
            V = int_V(ST['V'], _t, ST['input'])

            is_ref = _t - ST['t_last_spike'] < t_refractory
            V = np.where(is_ref, ST['V'], V)

            is_spike = V > V_th
            V[is_spike] = V_reset
            is_ref[is_spike] = 1.
            ST['t_last_spike'][is_spike] = _t
            
            ST['V'] = V
            ST['spike'] = is_spike
            ST['refractory'] = is_ref
            ST['input'] = 0.  # reset input here or it will be brought to next step

        return bp.NeuType(name='LIF_neuron',
                         ST=ST,
                         steps=update,
                         mode='vector')
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
