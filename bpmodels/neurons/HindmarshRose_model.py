# -*- coding: utf-8 -*-

import sys

import brainpy as bp


def get_HindmarshRose(a=1., b=3., c=1., d=5., r=0.01, s=4., V_rest=-1.6, noise = 0., mode='vector'):
    """
    Hindmarsh-Rose neuron model.

    .. math::
        &\\frac{d V}{d t} = y - a V^3 + b V^2 - z + I

        &\\frac{d y}{d t} = c - d V^2 - y

        &\\frac{d z}{d t} = r (s (V - V_{rest}) - z)
    
    **Neuron Parameters**
    
    ============= ============== ========= ============================================================
    **Parameter** **Init Value** **Unit**  **Explanation**
    ------------- -------------- --------- ------------------------------------------------------------
    a             1.             \         Model parameter. 
    
                                           Fixed to a value best fit neuron activity.

    b             3.             \         Model parameter. 
    
                                           Allows the model to switch between bursting

                                           and spiking, controls the spiking frequency.

    c             1.             \         Model parameter. 
    
                                           Fixed to a value best fit neuron activity.

    d             5.             \         Model parameter. 
    
                                           Fixed to a value best fit neuron activity.

    r             0.01           \         Model parameter. 
    
                                           Controls slow variable z's variation speed.
                                           
                                           Governs spiking frequency when spiking, and affects the 
                                           
                                           number of spikes per burst when bursting.

    s             4.             \         Model parameter. Governs adaption.

    V_rest        -1.6           \         Membrane resting potential.

    noise         0.             \         noise.

    mode          'scalar'       \         Data structure of ST members.
    ============= ============== ========= ============================================================

    Returns:
        bp.NeuType: return description of Hindmarsh-Rose neuron model.
       

    **Neuron State**
    
    =============== ================= =====================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- -------------------------------------
    V               -1.6              Membrane potential.
    
    y               -10.              Gating variable.
                             
    z               0.                Gating variable.
    
    input           0.                External and synaptic input current.
    =============== ================= =====================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    References:
        .. [1] Hindmarsh, James L., and R. M. Rose. "A model of neuronal bursting using 
               three coupled first order differential equations." Proceedings of the 
               Royal society of London. Series B. Biological sciences 221.1222 (1984): 
               87-102.
        .. [2] Storace, Marco, Daniele Linaro, and Enno de Lange. "The Hindmarsh–Rose 
               neuron model: bifurcation analysis and piecewise-linear approximations." 
               Chaos: An Interdisciplinary Journal of Nonlinear Science 18.3 (2008): 
               033128.
    """

    ST = bp.types.NeuState(
        {'V': -1.6, 'y': -10., 'z': 0., 'input': 0}
    )

    @bp.integrate
    def int_V(V, t, y, z, I_ext):
        return y - a * V * V * V + b * V * V - z + I_ext, noise

    @bp.integrate
    def int_y(y, t, V):
        return c - d * V * V - y

    @bp.integrate
    def int_z(z, t, V):
        return r * (s * (V - V_rest) - z)

    def update(ST, _t):
        V = int_V(ST['V'], _t, ST['y'], ST['z'], ST['input'])
        y = int_y(ST['y'], _t, ST['V'])
        z = int_z(ST['z'], _t, ST['V'])
        ST['V'] = V
        ST['y'] = y
        ST['z'] = z
        ST['input'] = 0

    if mode == 'scalar' or mode == 'vector':
        return bp.NeuType(name="HindmarshRose_neuron",
                          ST=ST,
                          steps=update,
                          mode=mode)
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
