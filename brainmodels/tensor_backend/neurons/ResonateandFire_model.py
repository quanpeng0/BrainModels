# -*- coding: utf-8 -*-

import brainpy as bp

bp.backend.set('numpy', dt=0.002)

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

    mode          'scalar'       \        Data structure of ST members.
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
        self.V = bp.backend.zeros(size)
        self.x = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)
        self.spike = bp.backend.zeros(size, dtype = bool)

        super(ResonateandFire, self).__init__(size = size, **kwargs)

    @staticmethod
    @bp.odeint()
    def integral(V, x, t, b, omega):
        dVdt = omega * x + b * V
        dxdt = b * x - omega * V
        return dVdt, dxdt

    def update(self, _t):
        x = self.x + self.input
        V, x = self.integral(self.V, x, _t, self.b, self.omega)
        sp = (V > self.V_th)
        V[sp] = self.V_reset
        x[sp] = self.x_reset
        self.V = V
        self.x = x
        self.spike = sp
        self.input[:] = 0