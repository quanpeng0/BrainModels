# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np


def get_LIF(V_rest=0., V_reset=-5., V_th=20., R=1.,
            tau=10., t_refractory=5., noise=0., mode='scalar'):
    """Leaky Integrate-and-Fire neuron model.
        
    .. math::

        \\tau \\frac{d V}{d t}=-(V-V_{rest}) + RI(t)

    **Neuron Parameters**






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
    
    Args:
        V_rest (float): Resting potential.
        V_reset (float): Reset potential after spike.
        V_th (float): Threshold potential of spike.
        R (float): Membrane resistance.
        C (float): Membrane capacitance.
        tau (float): Membrane time constant. Compute by R * C.
        t_refractory (int): Refractory period length.(ms)
        noise (float): noise.
        mode (str): Data structure of ST members.
        
    Returns:
        bp.Neutype: return description of LIF model.
        
    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """

    ST = bp.types.NeuState(
        {'V': 0, 'input': 0, 'spike': 0, 'refractory': 0, 't_last_spike': -1e7}
    )

    @bp.integrate
    def int_V(V, t, I_ext):  # integrate u(t)
        return (- (V - V_rest) + R * I_ext) / tau, noise / tau

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

    if mode == 'scalar':
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
            spike_idx = np.where(is_spike)[0]
            if len(spike_idx):
                V[spike_idx] = V_reset
                is_ref[spike_idx] = 1.
                ST['t_last_spike'][spike_idx] = _t
            ST['V'] = V
            ST['spike'] = is_spike
            ST['refractory'] = is_ref
            ST['input'] = 0.

        return bp.NeuType(name='LIF',
                         ST=ST,
                         steps=update,
                         mode='vector')
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
