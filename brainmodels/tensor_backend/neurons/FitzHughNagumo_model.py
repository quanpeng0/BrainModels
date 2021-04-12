# -*- coding: utf-8 -*-

import brainpy as bp


__all__ = [
    'FitzHughNagumo'
]


class FitzHughNagumo(bp.NeuGroup):
    """FitzHugh-Nagumo neuron model.

    """
    target_backend = 'general'

    @staticmethod
    def derivative(V, w, t, Iext, a, b, tau):
        dw = (V + a - b * w) / tau
        dV = V - V * V * V / 3 - w + Iext
        return dV, dw

    def __init__(self, size, a=0.7, b=0.8, tau=12.5, Vth=1.9, **kwargs):
        self.a = a
        self.b = b
        self.tau = tau
        self.Vth = Vth

        num = bp.size2len(size)
        self.V = bp.ops.zeros(num)
        self.w = bp.ops.zeros(num)
        self.spike = bp.ops.zeros(num, dtype=bool)
        self.input = bp.ops.zeros(num)

        self.integral = bp.odeint(self.derivative)
        super(FitzHughNagumo, self).__init__(size=size, **kwargs)

    def update(self, _t):
        V, self.w = self.integral(self.V, self.w, _t, self.input, self.a, self.b, self.tau)
        self.spike = (V >= self.Vth) * (self.V < self.Vth)
        self.V = V
        self.input[:] = 0.
