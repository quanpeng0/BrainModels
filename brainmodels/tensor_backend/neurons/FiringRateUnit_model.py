# -*- coding: utf-8 -*-
import brainpy as bp

bp.backend.set('numba', dt=0.01)


class FiringRateUnit(bp.NeuGroup):
    """
    Firing rate neuron model in Wilson-Cowan network

    Each neuron refers to a column of neurons, including excitatory and 
    inhibitory neurons.

    .. math::

        &\\tau_e\\frac{d a_e(t)}{d t} = 
            - a_e(t) + (k_e - r_e * a_e(t)) * 
                        \\mathcal{S}_e(c_1 a_e(t) - c_2 a_i(t) + I_{ext_e}(t))

        &\\tau_i\\frac{d a_i(t)}{d t} = 
            - a_i(t) + (k_i - r_i * a_i(t)) * 
                        \\mathcal{S}_i(c_3 a_e(t) - c_4 a_i(t) + I_{ext_j}(t))

        &\\mathcal{S}(x) = \\frac{1}{1 + exp(- a(x - \\theta))} - \\frac{1}{1 + exp(a\\theta)} 

    **Neuron Parameters**

    ============= ============== ======== ========================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------------------------------
    c1            12.            \        Weight from E-neurons to E-neurons.

    c2            4.             \        Weight from I-neurons to E-neurons.

    c3            13.            \        Weight from E-neurons to I-neurons.

    c4            11.            \        Weight from I-neurons to I-neurons.

    k_e           1.             \        Model parameter, control E-neurons' 

                                          refractory period together with r_e.

    k_i           1.             \        Model parameter, control I-neurons' 

                                          refractory period together with r_i.

    tau_e         1.             \        Time constant of E-neurons' activity.

    tau_i         1.             \        Time constant of I-neurons' activity.

    r_e           1.             \        Model parameter, control E-neurons' 

                                          refractory period together with k_e.

    r_i           1.             \        Model parameter, control I-neurons' 

                                          refractory period together with k_i.

    slope_e       1.2            \        E-neurons' sigmoid function slope parameter.

    slope_i       1.             \        I-neurons' sigmoid function slope parameter.

    theta_e       1.8            \        E-neurons' sigmoid function phase parameter.

    theta_i       4.             \        I-neurons' sigmoid function phase parameter.

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ========================================================================

    **Neuron Variables**    

    An object of neuron class record those variables for each neuron:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    a_e                0.1               The proportion of excitatory cells firing per unit time.

    a_i                0.05              The proportion of inhibitory cells firing per unit time.

    input_e            0.                External input to excitatory cells.

    input_i            0.                External input to inhibitory cells.
    ================== ================= =========================================================

    References:
        .. [1] Wilson, Hugh R., and Jack D. Cowan. "Excitatory and inhibitory 
               interactions in localized populations of model neurons." 
               Biophysical journal 12.1 (1972): 1-24.


    """
    target_backend = 'general'

    @staticmethod
    def derivative(a_e, a_i, t,
                   k_e, r_e, c1, c2, I_ext_e,
                   slope_e, theta_e, tau_e,
                   k_i, r_i, c3, c4, I_ext_i,
                   slope_i, theta_i, tau_i):
        daedt = (- a_e + (k_e - r_e * a_e) \
                 * mysigmoid(c1 * a_e - c2 * a_i + I_ext_e, slope_e, theta_e)) \
                / tau_e
        daidt = (- a_i + (k_i - r_i * a_i) \
                 * mysigmoid(c3 * a_e - c4 * a_i + I_ext_i, slope_i, theta_i)) \
                / tau_i
        return daedt, daidt

    def __init__(self, size, c1=12., c2=4., c3=13., c4=11.,
                 k_e=1., k_i=1., tau_e=1., tau_i=1., r_e=1., r_i=1.,
                 slope_e=1.2, slope_i=1., theta_e=2.8, theta_i=4.,
                 **kwargs):
        # params
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.k_e = k_e
        self.k_i = k_i
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.r_e = r_e
        self.r_i = r_i
        self.slope_e = slope_e
        self.slope_i = slope_i
        self.theta_e = theta_e
        self.theta_i = theta_i

        # vars
        self.input_e = bp.backend.zeros(size)
        self.input_i = bp.backend.zeros(size)
        self.a_e = bp.backend.ones(size) * 0.1
        self.a_i = bp.backend.ones(size) * 0.05

        self.integral = bp.odeint(self.derivative)
        super(FiringRateUnit, self).__init__(size=size, **kwargs)

    def mysigmoid(x, a, theta):
        return 1 / (1 + np.exp(- a * (x - theta))) \
               - 1 / (1 + np.exp(a * theta))

    def update(self, _t):
        self.a_e, self.a_i = self.integral(
            self.a_e, self.a_i, _t,
            self.k_e, self.r_e, self.c1, self.c2,
            self.input_e, self.slope_e,
            self.theta_e, self.tau_e,
            self.k_i, self.r_i, self.c3, self.c4,
            self.input_i, self.slope_i,
            self.theta_i, self.tau_i)
        self.input_e[:] = 0.
        self.input_i[:] = 0.
