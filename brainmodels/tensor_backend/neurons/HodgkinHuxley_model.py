# -*- coding: utf-8 -*-


import brainpy as bp

__all__ = [
    'HH'
]


class HH(bp.NeuGroup):
    """Hodgkin–Huxley neuron model.

    .. math::

        C \\frac {dV} {dt} = -(\\bar{g}_{Na} m^3 h (V &-E_{Na})
        + \\bar{g}_K n^4 (V-E_K) + g_{leak} (V - E_{leak})) + I(t)

        \\frac {dx} {dt} &= \\alpha_x (1-x)  - \\beta_x, \\quad x\\in {\\rm{\\{Na, K, leak\\}}}

        &\\alpha_m(V) = \\frac {0.1(V+40)}{1-exp(\\frac{-(V + 40)} {10})}

        &\\beta_m(V) = 4.0 exp(\\frac{-(V + 65)} {18})

        &\\alpha_h(V) = 0.07 exp(\\frac{-(V+65)}{20})

        &\\beta_h(V) = \\frac 1 {1 + exp(\\frac{-(V + 35)} {10})}

        &\\alpha_n(V) = \\frac {0.01(V+55)}{1-exp(-(V+55)/10)}

        &\\beta_n(V) = 0.125 exp(\\frac{-(V + 65)} {80})


    **Neuron Parameters**

    ============= ============== ======== ====================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------
    V_th          20.            mV       the spike threshold.

    C             1.             ufarad   capacitance.

    E_Na          50.            mV       reversal potential of sodium.

    E_K           -77.           mV       reversal potential of potassium.

    E_leak        54.387         mV       reversal potential of unspecific.

    g_Na          120.           msiemens conductance of sodium channel.

    g_K           36.            msiemens conductance of potassium channel.

    g_leak        .03            msiemens conductance of unspecific channels.

    noise         0.             \        the noise fluctuation.

    mode          'vector'       \        Data structure of ST members.
    ============= ============== ======== ====================================

    **Neuron State**

    ST refers to the neuron state, items in ST are listed below:

    =============== ==================  =========================================================
    **Member name** **Initial values**  **Explanation**
    --------------- ------------------  ---------------------------------------------------------
    V                        -65         Membrane potential.

    m                        0.05        gating variable of the sodium ion channel.

    n                        0.32        gating variable of the potassium ion channel.

    h                        0.60        gating variable of the sodium ion channel.

    input                     0          External and synaptic input current.

    spike                     0          Flag to mark whether the neuron is spiking.
                                         Can be seen as bool.
    =============== ==================  =========================================================

    Note that all ST members are saved as floating point type in BrainPy,
    though some of them represent other data types (such as boolean).

    References:
        .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description
               of membrane current and its application to conduction and excitation
               in nerve." The Journal of physiology 117.4 (1952): 500.

    """

    target_backend = 'general'

    @staticmethod
    def derivative(V, m, h, n, t, C, gNa, ENa, gK, EK, gL, EL, Iext):
        alpha = 0.1 * (V + 40) / (1 - bp.backend.exp(-(V + 40) / 10))
        beta = 4.0 * bp.backend.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m

        alpha = 0.07 * bp.backend.exp(-(V + 65) / 20.)
        beta = 1 / (1 + bp.backend.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h

        alpha = 0.01 * (V + 55) / (1 - bp.backend.exp(-(V + 55) / 10))
        beta = 0.125 * bp.backend.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n

        I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
        I_K = (gK * n ** 4.0) * (V - EK)
        I_leak = gL * (V - EL)
        dVdt = (- I_Na - I_K - I_leak + Iext) / C

        return dVdt, dmdt, dhdt, dndt

    def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36.,
                 EL=-54.387, gL=0.03, V_th=20., C=1.0, **kwargs):
        # parameters
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.C = C
        self.V_th = V_th

        # variables
        num = bp.size2len(size)
        self.V = -65. * bp.backend.ones(num)
        self.m = 0.5 * bp.backend.ones(num)
        self.h = 0.6 * bp.backend.ones(num)
        self.n = 0.32 * bp.backend.ones(num)
        self.spike = bp.backend.zeros(num, dtype=bool)
        self.input = bp.backend.zeros(num)

        # numerical solver
        self.integral = bp.odeint(f=self.derivative, method='exponential_euler')
        super(HH, self).__init__(size=size, **kwargs)

    def update(self, _t):
        V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t,
                                   self.C, self.gNa, self.ENa, self.gK,
                                   self.EK, self.gL, self.EL, self.input)
        self.spike = (self.V < self.V_th) * (V >= self.V_th)
        self.V = V
        self.m = m
        self.h = h
        self.n = n
        self.input[:] = 0
