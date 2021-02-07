# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np


def get_AdExIF(V_rest=-65., V_reset=-68., V_th=-30.,
               V_T=-59.9, delta_T=3.48, a=1., b=1,
               R=1, tau=10., tau_w=30.,
               t_refractory=0., noise=0., mode='scalar'):
    """Adaptive Exponential Integrate-and-Fire neuron model.
    
    .. math::
    
        \\tau_m\\frac{d V}{d t}= - (V-V_{rest}) + \\Delta_T e^{\\frac{V-V_T}{\\Delta_T}} - R w + RI(t)
    
        \\tau_w \\frac{d w}{d t}=a(V-V_{rest}) - w + b \\tau_w \\sum \\delta (t-t^f)


    **Neuron Parameters**

    ============= ============== ======== ========================================================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
    V_rest        -65.           mV       Resting potential.

    V_reset       -68.           mV       Reset potential after spike.

    V_th          -30.           mV       Threshold potential of spike and reset.

    V_T           -59.9          mV       Threshold potential of generating action potential.

    delta_T       3.48           \        Spike slope factor.

    a             1              \        The sensitivity of the recovery variable :math:`u` to the sub-threshold fluctuations of the membrane potential :math:`v`

    b             1              \        The increment of :math:`w` produced by a spike.

    R             1              \        Membrane resistance.

    tau           10             ms       Membrane time constant. Compute by R * C.

    tau_w         30             ms       Time constant of the adaptation current.

    t_refractory  0              ms       Refractory period length.

    noise         0.             \        the noise fluctuation.

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ========================================================================================================================

    Returns:
        bp.Neutype: return description of the AdExIF model.

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
        .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation 
               mechanisms determine the neuronal response to fluctuating 
               inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    """

    ST = bp.types.NeuState('V', 'w', 'input', 'spike', 'refractory', t_last_spike=-1e7)

    @bp.integrate
    def int_V(V, t, w, I_ext):  # integrate u(t)
        return (- (V - V_rest) + delta_T * np.exp((V - V_T) / delta_T) - R * w + R * I_ext) / tau, noise / tau

    @bp.integrate
    def int_w(w, t, V):
        return (a * (V - V_rest) - w) / tau_w, noise / tau_w

    if mode == 'scalar':
        def update(ST, _t):
            if _t - ST['t_last_spike'] <= t_refractory:
                ST['refractory'] = 1.
                ST['spike'] = 0.
            else:
                ST['refractory'] = 0.
                w = int_w(ST['w'], _t, ST['V'])
                V = int_V(ST['V'], _t, w, ST['input'])
                if V >= V_th:
                    V = V_reset
                    w += b
                    ST['spike'] = 1.
                    ST['t_last_spike'] = _t
                else:
                    ST['spike'] = 0.
                ST['V'] = V
                ST['w'] = w
            ST['input'] = 0.

    elif mode == 'vector':
        def update(ST, _t):
            w = int_w(ST['w'], _t, ST['V'])
            V = int_V(ST['V'], _t, w, ST['input'])
            is_ref = _t - ST['t_last_spike'] <= t_refractory
            V = np.where(is_ref, ST['V'], V)
            w = np.where(is_ref, ST['w'], w)

            is_spike = V > V_th
            V[is_spike] = V_reset
            w[is_spike] += b
            is_ref[is_spike] = 1.
            ST['t_last_spike'][is_spike] = _t

            ST['V'] = V
            ST['w'] = w
            ST['spike'] = is_spike
            ST['refractory'] = is_ref
            ST['input'] = 0.

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.NeuType(name='AdExIF_neuron',
                      ST=ST,
                      steps=update,
                      mode=mode)
