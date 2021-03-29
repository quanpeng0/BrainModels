# -*- coding: utf-8 -*-


import brainpy as bp
from numba import prange

__all__ = [
    'HindmarshRose'
]


class HindmarshRose(bp.NeuGroup):
    """Hindmarsh-Rose neuron model.

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

       noise         0.             \         noise.
       ============= ============== ========= ============================================================

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

    @staticmethod
    def derivative(V, y, z, t, a, b, c, d, r, s, x_r, Isyn):
        dV = y - a * V ** 3 + b * V * V - z + Isyn
        dy = c - d * V * V - y
        dz = r * (s * (V - x_r) - z)
        return dV, dy, dz

    def __init__(self, size, a=1., b=3., c=1., d=5., s=4., x_r=-1.6,
                 r=0.001, Vth=1.9, **kwargs):
        # parameters
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.s = s
        self.x_r = x_r
        self.r = r
        self.Vth = Vth

        # variables
        num = bp.size2len(size)
        self.V = -1.6 * bp.backend.ones(num)
        self.y = -10 * bp.backend.ones(num)
        self.z = bp.backend.zeros(num)
        self.input = bp.backend.zeros(num)
        self.spike = bp.backend.zeros(num, dtype=bool)

        self.integral = bp.odeint(f=self.derivative)

        super(HindmarshRose, self).__init__(size=size, **kwargs)

    def update(self, _t):
        for i in prange(self.num):
            V, y, z = self.integral(self.V[i], self.y[i], self.z[i], _t,
                                    self.a, self.b, self.c, self.d, self.r,
                                    self.s, self.x_r, self.input[i])
            self.spike[i] = (V >= self.Vth) * (self.V[i] < self.Vth)
            self.V[i] = V
            self.y[i] = y
            self.z[i] = z
            self.input[i] = 0.
