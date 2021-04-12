# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np

__all__ = [
    'MorrisLecar'
]


class MorrisLecar(bp.NeuGroup):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numba-cuda']

    @staticmethod
    def derivative(V, W, t, V1, V2, g_Ca, V_Ca, g_K, V_K,
                   g_leak, V_leak, C, I_ext, phi, V3, V4):
        M_inf = (1 / 2) * (1 + np.tanh((V - V1) / V2))
        I_Ca = g_Ca * M_inf * (V - V_Ca)
        I_K = g_K * W * (V - V_K)
        I_Leak = g_leak * (V - V_leak)
        dVdt = (- I_Ca - I_K - I_Leak + I_ext) / C
        tau_W = 1 / (phi * np.cosh((V - V3) / (2 * V4)))
        W_inf = (1 / 2) * (1 + np.tanh((V - V3) / V4))
        dWdt = (W_inf - W) / tau_W
        return dVdt, dWdt

    def __init__(self, size, V_Ca=130., g_Ca=4.4, V_K=-84., g_K=8.,
                 V_leak=-60., g_leak=2., C=20., V1=-1.2, V2=18.,
                 V3=2., V4=30., phi=0.04, **kwargs):
        # params
        self.V_Ca = V_Ca
        self.g_Ca = g_Ca
        self.V_K = V_K
        self.g_K = g_K
        self.V_leak = V_leak
        self.g_leak = g_leak
        self.C = C
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.V4 = V4
        self.phi = phi

        # vars
        num = bp.size2len(size)
        self.input = bp.ops.zeros(num)
        self.V = bp.ops.ones(num) * -20.
        self.W = bp.ops.ones(num) * 0.02

        self.integral = bp.odeint(f=self.derivative)
        super(MorrisLecar, self).__init__(size=size, **kwargs)

    def update(self, _t):
        self.V, self.W = self.integral(
            self.V, self.W, _t, self.V1, self.V2,
            self.g_Ca, self.V_Ca, self.g_K, self.V_K,
            self.g_leak, self.V_leak, self.C, self.input,
            self.phi, self.V3, self.V4
        )
        self.input[:] = 0.
