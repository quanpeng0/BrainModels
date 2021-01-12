# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np
import sys


def get_ExpIF(V_rest=-65., V_reset=-68., V_th=-30., V_T=-59.9, delta_T=3.48,
              R=10., C=1., tau=10., t_refractory=1.7, noise=0., mode='scalar'):
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

    noise         0.             \        noise.

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ===================================================

    Returns:
        bp.Neutype: return description of ExpIF model.

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
        .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation 
               mechanisms determine the neuronal response to fluctuating 
               inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    """

    ST = bp.types.NeuState('V', 'input', 'spike',
                           'refractory', t_last_spike=-1e7)

    @bp.integrate
    def int_V(V, t, I_ext):  # integrate u(t)
        return (- (V - V_rest) + delta_T * np.exp((V - V_T) / delta_T) + R * I_ext) / tau, noise / tau

    if mode == 'scalar':

        def update(ST, _t):
            # update variables
            ST['spike'] = 0
            ST['refractory'] = 1. if _t - \
                ST['t_last_spike'] <= t_refractory else 0.
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

    elif mode == 'vector':

        def update(ST, _t):
            V = int_V(ST['V'], _t, ST['input'])

            is_ref = _t - ST['t_last_spike'] < t_refractory
            V = np.where(is_ref, ST['V'], V)
            is_spike = V > V_th

            V[is_spike] = V_reset
            is_ref[is_spike] = 1.
            ST['t_last_spike'][is_spike] = _t

            ST['refractory'] = is_ref
            ST['spike'] = is_spike
            ST['V'] = V
            ST['input'] = 0  # reset input here or it will be brought to next step
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.NeuType(name='ExpIF_neuron',
                      ST=ST,
                      steps=update,
                      mode=mode)
