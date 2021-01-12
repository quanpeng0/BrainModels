# -*- coding: utf-8 -*-
import sys

import brainpy as bp
import numpy as np


def get_MorrisLecar(noise=0., V_Ca=130., g_Ca=4.4, V_K=-84., g_K=8., V_Leak=-60.,
                    g_Leak=2., C=20., V1=-1.2, V2=18., V3=2., V4=30., phi=0.04, mode='vector'):
    """
    The Morris-Lecar neuron model. (Also known as :math:`I_{Ca}+I_K`-model.)

    .. math::

        C\\frac{dV}{dt} = -  g_{Ca} M_{\\infty} (V - & V_{Ca})- g_{K} W(V - V_{K}) - g_{Leak} (V - V_{Leak}) + I_{ext}

        & \\frac{dW}{dt} = \\frac{W_{\\infty}(V) - W}{ \\tau_W(V)} 

    **Neuron Parameters**
    
    ============= ============== ======== =======================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -------------------------------------------------------
    noise         0.             \        The noise fluctuation.
    V_Ca          130.           mV       Equilibrium potentials of Ca+.(mV)
    g_Ca          4.4            \        Maximum conductance of corresponding Ca+.(mS/cm2)
    V_K           -84.           mV       Equilibrium potentials of K+.(mV)
    g_K           8.             \        Maximum conductance of corresponding K+.(mS/cm2)
    V_Leak        -60.           mV       Equilibrium potentials of leak current.(mV)
    g_Leak        2.             \        Maximum conductance of leak current.(mS/cm2)
    C             20.            \        Membrane capacitance.(uF/cm2)
    V1            -1.2           \        Potential at which M_inf = 0.5.(mV)
    V2            18.            \        Reciprocal of slope of voltage dependence of M_inf.(mV)
    V3            2.             \        Potential at which W_inf = 0.5.(mV)
    V4            30.            \        Reciprocal of slope of voltage dependence of W_inf.(mV)
    phi           0.04           \        A temperature factor.(1/s)
    mode          'vector'       \        Data structure of ST members.
    ============= ============== ======== =======================================================
    
    Returns:
        bp.Neutype: return description of Morris-Lecar model.


    **Neuron State**
    
    ST refers to neuron state, members of ST are listed below:
    
    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    V               -20.              Membrane potential.
    
    W               0.02              Gating variable, refers to the fraction of 
                                      opened K+ channels.
    
    input           0.                External and synaptic input current.
    =============== ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    References:
        .. [1] Meier, Stephen R., Jarrett L. Lancaster, and Joseph M. Starobin.
               "Bursting regimes in a reaction-diffusion system with action 
               potential-dependent equilibrium." PloS one 10.3 (2015): 
               e0122401.
    """

    ST = bp.types.NeuState('input', V=-20., W=0.02)

    @bp.integrate
    def int_W(W, t, V):
        tau_W = 1 / (phi * np.cosh((V - V3) / (2 * V4)))
        W_inf = (1 / 2) * (1 + np.tanh((V - V3) / V4))
        dWdt = (W_inf - W) / tau_W
        return dWdt

    @bp.integrate
    def int_V(V, t, W, Isyn):
        M_inf = (1 / 2) * (1 + np.tanh((V - V1) / V2))
        I_Ca = g_Ca * M_inf * (V - V_Ca)
        I_K = g_K * W * (V - V_K)
        I_Leak = g_Leak * (V - V_Leak)
        dVdt = (- I_Ca - I_K - I_Leak + Isyn) / C
        return dVdt, noise / C

 
    if mode == 'scalar' or mode == 'vector':
        def update(ST, _t):
            W = int_W(ST['W'], _t, ST['V'])
            V = int_V(ST['V'], _t, ST['W'], ST['input'])
            ST['V'] = V
            ST['W'] = W
            ST['input'] = 0.
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
    
    return bp.NeuType(name='MorrisLecar_neuron',
                      ST=ST,
                      steps=update,
                      mode=mode)
