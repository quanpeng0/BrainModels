# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'IL',
]


class IL(bp.Channel):
  def __init__(self, g_max, E, **kwargs):
    super(IL, self).__init__(**kwargs)

    self.E = E
    self.g_max = g_max

  def init(self, host):
    super(IL, self).init(host)
    self.I = bp.math.Variable(bp.math.zeros(host.num, dtype=bp.math.float_))

  def update(self, _t, _dt):
    self.I[:] = self.g_max * (self.E - self.host.V)


class IKL(IL):
  pass


