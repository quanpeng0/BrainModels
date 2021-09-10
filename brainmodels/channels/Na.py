# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'INa',
]


class INa(bp.Channel):
  def __init__(self, E=50., g_max=120., **kwargs):
    super(INa, self).__init__(**kwargs)

    self.E = E
    self.g_max = g_max

  def init(self, host, ):
    super(INa, self).init(host)
    self.m = bp.math.Variable(bp.math.zeros(host.num, dtype=bp.math.float_))
    self.h = bp.math.Variable(bp.math.zeros(host.num, dtype=bp.math.float_))
    self.I = bp.math.Variable(bp.math.zeros(host.num, dtype=bp.math.float_))

  @bp.odeint(method='exponential_euler')
  def integral(self, m, h, t, V):
    alpha = 0.1 * (V + 40) / (1 - bp.math.exp(-(V + 40) / 10))
    beta = 4.0 * bp.math.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m

    alpha = 0.07 * bp.math.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bp.math.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h

    return dmdt, dhdt

  def update(self, _t, _dt):
    self.m[:], self.h[:] = self.integral(self.m, self.h, _t, self.host.V, dt=_dt)
    self.I[:] = (self.g_max * self.m ** 3 * self.h) * (self.host.V - self.E)


