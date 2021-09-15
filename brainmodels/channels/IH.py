# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm

__all__ = [
  'Ih',
]


class Ih(bp.Channel):
  r"""The hyperpolarization-activated cation current model.

  The hyperpolarization-activated cation current model is adopted from (Huguenard, et, al., 1992) [1]_.
  Its dynamics is given by:

  .. math::

      \begin{aligned}
      I_h &= g_{\mathrm{max}} p
      \\
      \frac{dp}{dt} &= \phi \frac{p_{\infty} - p}{\tau_p}
      \\
      p_{\infty} &=\frac{1}{1+\exp ((V+75) / 5.5)}
      \\
      \tau_{p} &=\frac{1}{\exp (-0.086 V-14.59)+\exp (0.0701 V-1.87)}
      \end{aligned}

  where :math:`\phi=1` is a temperature-dependent factor.

  Parameters
  ----------
  g_max : float
    The maximal conductance density (:math:`mS/cm^2`).
  E : float
    The reversal potential (mV).
  phi : float
    The temperature-dependent factor.

  References
  ----------
  .. [1] Huguenard, John R., and David A. McCormick. "Simulation of the currents involved in rhythmic oscillations in thalamic relay neurons." Journal of neurophysiology 68, no. 4 (1992): 1373-1383.

  """

  def __init__(self, g_max=10., E=-90., phi=1., **kwargs):
    super(Ih, self).__init__(**kwargs)

    self.phi = phi
    self.g_max = g_max
    self.E = E

  def init(self, host, ):
    super(Ih, self).init(host)
    self.p = bp.math.Variable(bp.math.zeros(host.num, dtype=bp.math.float_))

  @bp.odeint(method='exponential_euler')
  def integral(self, p, t, V):
    p_inf = 1. / (1. + bm.exp((V + 75.) / 5.5))
    p_tau = 1. / (bm.exp(-0.086 * V - 14.59) + bm.exp(0.0701 * V - 1.87))
    dpdt = self.phi * (p_inf - p) / p_tau
    return dpdt

  def update(self, _t, _dt):
    self.p[:] = self.integral(self.p, _t, self.host.V, dt=_dt)
    self.host.input += (self.g_max * self.p) * (self.E - self.host.V)
