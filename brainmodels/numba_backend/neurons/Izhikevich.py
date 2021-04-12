# -*- coding: utf-8 -*-

import brainpy as bp
from numba import prange

__all__ = [
    'Izhikevich'
]


class Izhikevich(bp.NeuGroup):
    '''
    The Izhikevich neuron model.

    .. math ::

        \\frac{d V}{d t} &= 0.04 V^{2}+5 V+140-u+I

        \\frac{d u}{d t} &=a(b V-u)

    .. math ::

        \\text{if}  v \\geq 30  \\text{mV}, \\text{then}
        \\begin{cases} v \\leftarrow c \\\\ u \\leftarrow u+d \\end{cases}

    **Neuron Parameters**

    ============= ============== ======== ================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------------------------------------------------------
    type          None           \        The neuron spiking type.

    a             0.02           \        It determines the time scale of the recovery variable :math:`u`.

    b             0.2            \        It describes the sensitivity of the recovery variable :math:`u` to

                                          the sub-threshold fluctuations of the membrane potential :math:`v`.

    c             -65.           \        It describes the after-spike reset value of the membrane

                                          potential :math:`v` caused by the fast high-threshold :math:`K^{+}` conductance.

    d             8.             \        It describes after-spike reset of the recovery variable :math:`u`

                                          caused by slow high-threshold :math:`Na^{+}` and :math:`K^{+}` conductance.

    t_refractory  0.             ms       Refractory period length. [ms]

    V_th          30.            mV       The membrane potential threshold.
    ============= ============== ======== ================================================================================

    **Neuron Variables**

    An object of neuron class record those variables for each neuron:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                  float            -65        Membrane potential.

    u                  float            1          Recovery variable.

    input              float            0          External and synaptic input current.

    spike              float            0          Flag to mark whether the neuron is spiking.

                                                   Can be seen as bool.

    t_last_spike       float            -1e7       Last spike time stamp.
    ================== ======== ================== ===========================================

    References:
        .. [1] Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE
               Transactions on neural networks 14.6 (2003): 1569-1572.

        .. [2] Izhikevich, Eugene M. "Which model to use for cortical spiking neurons?."
               IEEE transactions on neural networks 15.5 (2004): 1063-1070.
    '''
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numba-cuda']

    @staticmethod
    def derivative(V, u, t, I_ext, a, b):
        dVdt = 0.04 * V * V + 5 * V + 140 - u + I_ext
        dudt = a * (b * V - u)
        return dVdt, dudt

    def __init__(self, size, a=0.02, b=0.20, c=-65., d=8.,
                 t_refractory=0., V_th=30., **kwargs):
        # params
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.t_refractory = t_refractory
        self.V_th = V_th

        # vars
        num = bp.size2len(size)
        self.V = bp.ops.ones(num) * -65.
        self.u = bp.ops.ones(num) * 1.
        self.input = bp.ops.zeros(num)
        self.spike = bp.ops.zeros(num, dtype=bool)
        self.refractory = bp.ops.zeros(num, dtype=bool)
        self.t_last_spike = bp.ops.ones(num) * -1e7

        self.integral = bp.odeint(self.derivative)
        super(Izhikevich, self).__init__(size=size, **kwargs)

    def update(self, _t):
        for i in prange(self.size[0]):
            spike = 0.
            refractory = (_t - self.t_last_spike[i] <= self.t_refractory)
            if not refractory:
                V, u = self.integral(
                    self.V[i], self.u[i], _t,
                    self.input[i], self.a, self.b)
                spike = (V >= self.V_th)
                if spike:
                    V = self.c
                    u += self.d
                    self.t_last_spike[i] = _t
                self.V[i] = V
                self.u[i] = u
            self.spike[i] = spike
            self.refractory[i] = refractory | spike
            self.input[i] = 0.


'''
        def update(ST, _t):
            V = int_V(ST['V'], _t, ST['u'], ST['input'])
            u = int_u(ST['u'], _t, ST['V'])

            is_ref = _t - ST['t_last_spike'] <= t_refractory
            V = np.where(is_ref, ST['V'], V)
            u = np.where(is_ref, ST['u'], u)

            is_spike = V > V_th
            V[is_spike] = c
            u[is_spike] += d
            is_ref[is_spike] = 1.
            ST['t_last_spike'][is_spike] = _t

            ST['V'] = V
            ST['u'] = u
            ST['spike'] = is_spike
            ST['input'] = 0.  # reset input here or it will be brought to next step
'''
