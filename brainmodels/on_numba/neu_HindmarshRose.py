# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
    'HindmarshRose'
]


class HindmarshRose(bp.NeuGroup):
    """Hindmarsh-Rose neuron model [1]_ [2]_.

    The Hindmarsh–Rose model of neuronal activity
    is aimed to study the spiking-bursting behavior
    of the membrane potential observed in experiments
    made with a single neuron.

    The model has the mathematical form of a system of
    three nonlinear ordinary differential equations on
    the dimensionless dynamical variables :math:`x(t)`,
    :math:`y(t)`, and :math:`z(t)`. They read:

       .. math::
           &\\frac{d V}{d t} = y - a V^3 + b V^2 - z + I

           &\\frac{d y}{d t} = c - d V^2 - y

           &\\frac{d z}{d t} = r (s (V - V_{rest}) - z)

    where :math:`a, b, c, d` model the working of the fast ion channels,
    :math:`I` models the slow ion channels.

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

   spike           False             Whether generate the spikes.

   input           0.                External and synaptic input current.

   t_last_spike       -1e7               Last spike time stamp.
   =============== ================= =====================================

   References
   ----------

   .. [1] Hindmarsh, James L., and R. M. Rose. "A model of neuronal bursting using
          three coupled first order differential equations." Proceedings of the
          Royal society of London. Series B. Biological sciences 221.1222 (1984):
          87-102.
   .. [2] Storace, Marco, Daniele Linaro, and Enno de Lange. "The Hindmarsh–Rose
          neuron model: bifurcation analysis and piecewise-linear approximations."
          Chaos: An Interdisciplinary Journal of Nonlinear Science 18.3 (2008):
          033128.
   """

    target_backend = ['numpy', 'numba']

    @staticmethod
    def derivative(V, y, z, t, a, b, I_ext, c, d, r, s, V_rest):
        dVdt = y - a * V * V * V + b * V * V - z + I_ext
        dydt = c - d * V * V - y
        dzdt = r * (s * (V - V_rest) - z)
        return dVdt, dydt, dzdt

    def __init__(self, size, a=1., b=3., c=1., d=5., r=0.01, s=4.,
                 V_rest=-1.6, V_th=1.0, **kwargs):
        super(HindmarshRose, self).__init__(size=size, **kwargs)

        # parameters
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.r = r
        self.s = s
        self.V_th = V_th
        self.V_rest = V_rest

        # variables
        self.z = bp.ops.zeros(self.num)
        self.V = bp.ops.ones(self.num) * -1.6
        self.y = bp.ops.ones(self.num) * -10.
        self.input = bp.ops.zeros(self.num)
        self.spike = bp.ops.zeros(self.num, dtype=bool)
        self.t_last_spike = bp.ops.ones(self.num) * -1e7

        self.integral = bp.odeint(f=self.derivative)

    def update(self, _t, _i, _dt):
        for i in range(self.num):
            V, self.y[i], self.z[i] = self.integral(
                self.V[i], self.y[i], self.z[i], _t,
                self.a, self.b, self.input[i],
                self.c, self.d, self.r, self.s,
                self.V_rest)
            spike = (V > self.V_th) and (V <= self.V_th)
            self.spike[i] = spike
            if spike:
                self.t_last_spike[i] = _t
            self.V[i] = V
            self.input[i] = 0.
