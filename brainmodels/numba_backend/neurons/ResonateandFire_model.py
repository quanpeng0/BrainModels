# -*- coding: utf-8 -*-

import brainpy as bp
from numba import prange

class ResonateandFire(bp.NeuGroup):
    """Resonate-and-fire neuron model.

    .. math::

        \\frac{d x}{d t} = b x - \\omega y

        \\frac{d y}{d t} = \\omega x + b y

    When spike,

    .. math::

        x \\leftarrow 0

        y \\leftarrow 1

    Or we can write the equations in equivalent complex form:    

    .. math::

        \\frac{d z}{d t} = (b + i \\omega) z

        z = x + i y \\in \\mathbb{C}


    When spike,

    .. math::

        z \\leftarrow i

    **Neuron Parameters**

    ============= ============== ======== ========================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------------------------------
    b             -1.            \        Parameter, refers to the rate of attrsction to the rest.

    omega         10.            \        Parameter. refers to the frequency of the oscillations.

    V_th          1.             \        Threshold potential of spike.

    V_reset       1.             \        Reset value for voltage-like variable after spike.

    x_reset       0.             \        Reset value for current-like variable after spike.
    ============= ============== ======== ========================================================

    **Neuron Variables**    

    An object of neuron class record those variables for each neuron:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                  0.                Voltage-like variable.

    x                  0.                Current-like variable.

    input              0.                External and synaptic input current.

    spike              0.                Flag to mark whether the neuron is spiking. 

                                         Can be seen as bool.

    t_last_spike       -1e7              Last spike time stamp.
    ================== ================= ==============================================

    References:
        .. [1] Izhikevich, Eugene M. "Resonate-and-fire neurons." 
               Neural networks 14.6-7 (2001): 883-894.

    """

    target_backend = 'general'

    @staticmethod
    def derivative(V, x, t, b, omega):
        dVdt = omega * x + b * V
        dxdt = b * x - omega * V
        return dVdt, dxdt

    def __init__(self, size, b=-1., omega=10., 
                 V_th=1., V_reset=1., x_reset=0.,
                 **kwargs):
        #parameters
        self.b = b
        self.omega = omega
        self.V_th = V_th
        self.V_reset = V_reset
        self.x_reset = x_reset

        #variables
        self.V = bp.ops.zeros(size)
        self.x = bp.ops.zeros(size)
        self.input = bp.ops.zeros(size)
        self.spike = bp.ops.zeros(size, dtype = bool)

        self.integral = bp.odeint(self.derivative)
        super(ResonateandFire, self).__init__(size = size, **kwargs)

    def update(self, _t):
        for i in prange(self.size[0]):
            x = self.x[i] + self.input[i]
            V, x = self.integral(self.V[i], x, _t, self.b, self.omega)
            self.spike[i]= (V > self.V_th)
            if self.spike[i]:
                V = self.V_reset
                x = self.x_reset
            self.V[i] = V
            self.x[i] = x
        self.input[:] = 0
