# -*- coding: utf-8 -*-

import brainpy as bp

bp.backend.set('numba', dt = 0.02)

class HindmarshRose(bp.NeuGroup):
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
    target_backend = 'general'

    def __init__(self, size, a=1., b=3., 
                 c=1., d=5., r=0.01, s=4., 
                 V_rest=-1.6, **kwargs):
        # parameters
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.r = r
        self.s = s
        self.V_rest = V_rest

        #variables
        self.z = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)
        self.V = bp.backend.ones(size) * -1.6
        self.y = bp.backend.ones(size) * -10.

        super(HindmarshRose, self).__init__(size = size, **kwargs)

    @staticmethod
    @bp.odeint()
    def integral(V, y, z, t, a, b, I_ext, c, d, r, s, V_rest):
        dVdt = y - a * V * V * V + b * V * V - z + I_ext
        dydt = c - d * V * V - y
        dzdt = r * (s * (V - V_rest) - z)
        return dVdt, dydt, dzdt

    def update(self, _t):
        V, y, z = self.integral(self.V, self.y, self.z, _t, 
                                self.a, self.b, self.input, 
                                self.c, self.d, self.r, self.s, 
                                self.V_rest)
        self.V = V
        self.y = y
        self.z = z
        self.input[:] = 0.

if __name__ == "__main__":
    mode = 'irregular_bursting'
    param= {'quiescence':         [1.0, 2.0],  #a
            'spiking':            [3.5, 5.0],  #c
            'bursting':           [2.5, 3.0],  #d
            'irregular_spiking':  [2.95, 3.3], #h
            'irregular_bursting': [2.8, 3.7],  #g
            }  
    #set params of b and I_ext corresponding to different firing mode
    print(f"parameters is set to firing mode <{mode}>")

    group = HindmarshRose(size = 10, b = param[mode][0],
                          monitors=['V'])

    group.run(350., inputs=('input', param[mode][1]), report=True)
    bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)