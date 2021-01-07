# -*- coding: utf-8 -*-

import sys

import brainpy as bp


def get_AdQuaIF(a=1, b=.1, a_0=.07,
                V_c=-50, V_rest=-65., V_reset=-68., V_th=-30.,
                R=1., tau=10., tau_w=10.,
                t_refractory=0., noise=0., mode='scalar'):
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

    ST = bp.types.NeuState('V', 'w', 'input', 'spike', 'refractory', t_last_spike = -1e7)
    

    @bp.integrate
    def int_V(V, t, w, I_ext):
        return (a_0* (V - V_rest)*(V-V_c) - R * w + R * I_ext) / tau, noise / tau

    @bp.integrate
    def int_w(w, t, V):
        return (a* (V - V_rest)-w) / tau_w, noise / tau_w

    if mode=='scalar':
        def update(ST, _t):
            ST['spike'] = 0
            if _t - ST['t_last_spike'] <= t_refractory:
                ST['refractory'] = 1.
            else:
                ST['refractory'] = 0.
                w = int_w(ST['w'], _t, ST['V'])
                V = int_V(ST['V'], _t, w, ST['input'])
                if V >= V_th:
                    V = V_reset
                    w += b
                    ST['spike'] = 1
                    ST['t_last_spike'] = _t
                ST['V'] = V
                ST['w'] = w
            # reset input
            ST['input'] = 0.
    elif mode == 'vector':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.NeuType(name='AdQuaIF_neuron',
                      ST=ST,
                      steps=update,
                      mode=mode)    
