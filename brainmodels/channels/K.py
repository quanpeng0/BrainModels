# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'IK',
]


class IK(bp.Channel):
  def __init__(self, E, g_max, **kwargs):
    super(IK, self).__init__(**kwargs)

    self.E = E
    self.g_max = g_max

  def init(self, host):
    super(IK, self).init(host)
    self.n = bp.math.Variable(bp.math.zeros(host.num, dtype=bp.math.float_))
    self.I = bp.math.Variable(bp.math.zeros(host.num, dtype=bp.math.float_))

  @bp.odeint(method='exponential_euler')
  def integral(self, n, t, V):
    alpha = 0.01 * (V + 55) / (1 - bp.math.exp(-(V + 55) / 10))
    beta = 0.125 * bp.math.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n
    return dndt

  def update(self, _t, _dt):
    self.n[:] = self.integral(self.n, _t, self.host.V, dt=_dt)
    self.I[:] = (self.g_max * self.n ** 4) * (self.host.V - self.E)
