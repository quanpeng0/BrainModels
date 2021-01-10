# -*- coding: utf-8 -*-

import sys
import numpy as np
import brainpy as bp


def get_GeneralizedIF(V_rest=-70., V_reset=-70.,
                      V_th_inf=-50., V_th_reset=-60.,
                      R=20., tau=20., a=0., b=0.01,
                      k1=0.2, k2=0.02, R1=0., R2=1., A1=0., A2=0.,
                      noise = 0.,
                      mode='scalar'):
    """
    Generalized Integrate-and-Fire model (GeneralizedIF model).
    
    .. math::
    
        &\\frac{d I_j}{d t} = - k_j I_j
    
        &\\frac{d V}{d t} = ( - (V - V_{rest}) + R\\sum_{j}I_j + RI) / \\tau
    
        &\\frac{d V_{th}}{d t} = a(V - V_{rest}) - b(V_{th} - V_{th\\infty})
    
    When V meet Vth, Generalized IF neuron fire:
    
    .. math::
    
        &I_j \\leftarrow R_j I_j + A_j
    
        &V \\leftarrow V_{reset}
    
        &V_{th} \\leftarrow max(V_{th_{reset}}, V_{th})
    
    Note that I_j refers to arbitrary number of internal currents.
    
    **Neuron Parameters**
    
    ============= ============== ======== ====================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------------------------------------------
    V_rest        -70.           mV       Resting potential.

    V_reset       -70.           mV       Reset potential after spike.

    V_th_inf      -50.           mV       Target value of threshold potential V_th updating.

    V_th_reset    -60.           mV       Free parameter, should be larger than V_reset.

    R             20.            \        Membrane resistance.

    tau           20.            \        Membrane time constant. Compute by R * C.

    a             0.             \        Coefficient describes the dependence of 
    
                                          V_th on membrane potential.

    b             0.01           \        Coefficient describes V_th update.

    k1            0.2            \        Constant pf I1.

    k2            0.02           \        Constant of I2.

    R1            0.             \        Free parameter. 
    
                                          Describes dependence of I_1 reset value on 
                                          
                                          I_1 value before spiking.

    R2            1.             \        Free parameter. 
    
                                          Describes dependence of I_2 reset value on 
                                          
                                          I_2 value before spiking.

    A1            0.             \        Free parameter.

    A2            0.             \        Free parameter.

    noise         0.             \        noise.

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ====================================================================
        
    Returns:
        bp.Neutype: return description of Generalized IF model.
        
    
    **Neuron State**
        
    ST refers to neuron state, members of ST are listed below:
    
    =============== ================= ==============================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ----------------------------------------------
    V               -70.              Membrane potential.
    
    input           0.                External and synaptic input current.
    
    spike           0.                Flag to mark whether the neuron is spiking. 
    
                                      Can be seen as bool.
    
    V_th            -50.              Spiking threshold potential.
                             
    I1              0.                Internal current 1.
    
    I2              0.                Internal current 2.
                             
    t_last_spike    -1e7              Last spike time stamp.
    =============== ================= ==============================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).
        
    References:
        .. [1] Mihalaş, Ştefan, and Ernst Niebur. "A generalized linear 
               integrate-and-fire neural model produces diverse spiking 
               behaviors." Neural computation 21.3 (2009): 704-718.
    """

    ST = bp.types.NeuState(
        {'V': -70., 'input': 0., 'spike': 0., 'V_th': -50.,
         'I1': 0., 'I2': 0.}
    )

    @bp.integrate
    def int_I1(I1, t):
        return - k1 * I1

    @bp.integrate
    def int_I2(I2, t):
        return - k2 * I2

    @bp.integrate
    def int_V_th(V_th, t, V):
        return a * (V - V_rest) - b * (V_th - V_th_inf)

    @bp.integrate
    def int_V(V, t, I_ext, I1, I2):
        return (- (V - V_rest) + R * I_ext + R * I1 + R * I2) / tau, noise / tau


    if mode == 'scalar':

        def update(ST, _t):
            ST['spike'] = 0
            I1 = int_I1(ST['I1'], _t)
            I2 = int_I2(ST['I2'], _t)
            V_th = int_V_th(ST['V_th'], _t, ST['V'])
            V = int_V(ST['V'], _t, ST['input'], ST['I1'], ST['I2'])
            if V > ST['V_th']:
                V = V_reset
                I1 = R1 * I1 + A1
                I2 = R2 * I2 + A2
                V_th = max(V_th, V_th_reset)
                ST['spike'] = 1
            ST['I1'] = I1
            ST['I2'] = I2
            ST['V_th'] = V_th
            ST['V'] = V
            ST['input'] = 0.

        return bp.NeuType(name='GeneralizedIF_neuron',
                          ST=ST,
                          steps=update,
                          mode=mode)
    elif mode == 'vector':

        def update(ST, _t):
            V = int_V(ST['V'], _t, ST['input'], ST['I1'], ST['I2'])
            V_th = int_V_th(ST['V_th'], _t, ST['V'])
            I1 = int_I1(ST['I1'], _t)
            I2 = int_I2(ST['I2'], _t)
            is_spike = V > ST['V_th']

            is_V_th_reset = np.logical_and(V_th < V_th_reset, is_spike)
            V[is_spike] = V_reset
            V_th = np.where(is_V_th_reset, V_th_reset,V_th)
            I1[is_spike] = R1 * I1[is_spike] + A1
            I2[is_spike] = R2 * I2[is_spike] + A2

            ST['spike'] = is_spike
            ST['I1'] = I1
            ST['I2'] = I2
            ST['V_th'] = V_th
            ST['V'] = V
            ST['input'] = 0.

        return bp.NeuType(name='GeneralizedIF_neuron',
                          ST=ST,
                          steps=update,
                          mode=mode)
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
