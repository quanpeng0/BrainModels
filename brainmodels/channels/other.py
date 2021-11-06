# -*- coding: utf-8 -*-

from .base import Channel

__all__ = [
  'IL',
  'IKL',
]


class IL(Channel):
  """The leakage channel current.

  Parameters
  ----------
  g_max : float
    The leakage conductance.
  E : float
    The reversal potential.
  """
  def __init__(self, g_max=0.1, E=-70., **kwargs):
    super(IL, self).__init__(**kwargs)

    self.E = E
    self.g_max = g_max

  def init(self, host, **kwargs):
    super(IL, self).init(host)

  def update(self, _t, _dt, **kwargs):
    self.host.I_ion += self.g_max * (self.E - self.host.V)
    self.host.V_linear -= self.g_max


class IKL(IL):
  """The potassium leak channel current.

  Parameters
  ----------
  g_max : float
    The potassium leakage conductance which is modulated by both
    acetylcholine and norepinephrine.
  E : float
    The reversal potential.
  """
  def __init__(self, g_max=0.005, E=-90., **kwargs):
    super(IKL, self).__init__(g_max=g_max, E=E, **kwargs)


