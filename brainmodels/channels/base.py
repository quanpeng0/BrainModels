# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
from brainpy import errors
from brainpy.base import Collector
from brainpy.simulation import brainobjects

from brainmodels.neurons.base import Neuron


__all__ = [
  'cond_neuron',
  'CondNeuGroup',
  'Channel',
  'MolChannel',
  'IonChannel',
]


def cond_neuron(**channels):
  pass


class CondNeuGroup(Neuron):
  """Base class to model conductance-based neuron group.

  The standard formulation for a conductance-based model is given as

  .. math::

      C_m {dV \over dt} = \sum_jg_j(E - V) + I_{ext}

  where :math:`g_j=\bar{g}_{j} M^x N^y` is the channel conductance, :math:`E` is the
  reversal potential, :math:`M` is the activation variable, and :math:`N` is the
  inactivation variable.

  :math:`M` and :math:`N` have the dynamics of

  .. math::

      {dx \over dt} = \phi_x {x_\infty (V) - x \over \tau_x(V)}

  where :math:`x \in [M, N]`, :math:`\phi_x` is a temperature-dependent factor,
  :math:`x_\infty` is the steady state, and :math:`\tau_x` is the time constant.
  Equivalently, the above equation can be written as:

  .. math::

      \frac{d x}{d t}=\phi_{x}\left(\alpha_{x}(1-x)-\beta_{x} x\right)

  where :math:`\alpha_{x}` and :math:`\beta_{x}` are rate constants.

  Parameters
  ----------
  size : int, tuple of int
    The network size of this neuron group.
  num_batch : optional, int
    The batch size.
  monitors : optional, list of str, tuple of str
    The neuron group monitor.
  name : optional, str
    The neuron group name.

  Notes
  -----

  The ``__init__()`` function in :py:class:`CondNeuGroup` is used to specify
  the parameters of channels and this neuron group. The ``__call__()`` function
  is used to initialize the variables in this neuron group.
  """
  target_backend = 'jax'

  def __init__(self, size, C=1., A=1e-3, V_th=0., method='euler',
               monitors=None, steps=('update',), name=None, **channels):
    super(CondNeuGroup, self).__init__(size, method=method, steps=steps,
                                       monitors=monitors, name=name)

    # parameters for neurons
    self.C = C
    self.A = A
    self.V_th = V_th

    # check 'channels'
    _channels = dict()
    for key in channels.items():
      assert isinstance(key, str), f'Key must be a str, but got {type(key)}: {key}'
      item = channels[key]
      assert isinstance(item, (tuple, list)) and len(item) == 2
      assert isinstance(item[1], dict)
      channels[key][1]['host'] = self
      channels[key][1]['method'] = method

    # initialize children channels
    self.channels = Collector()
    for key, (ch, params) in channels.items():
      self.channels[key] = ch(**params)
      if not isinstance(self.channels[key], Channel):
        raise errors.BrainPyError(f'{self.__class__.__name__} only receives {Channel} instance, '
                                  f'while we got {type(self.channels[key])}: {self.channels[key]}.')

  def update(self, _t, _dt, **kwargs):
    # update variables in channels
    for ch in self.child_channels.values():
      ch.update(_t, _dt)

    # update V using Exponential Euler method
    dvdt = self.I_ion / self.C + self.input * (1e-3 / self.A / self.C)
    linear = self.V_linear / self.C
    V = self.V + dvdt * (bm.exp(linear * _dt) - 1) / linear

    # update other variables
    self.spike[:] = bm.logical_and(V >= self.V_th, self.V < self.V_th)
    self.V_linear[:] = 0.
    self.I_ion[:] = 0.
    self.input[:] = 0.
    self.V[:] = V

  def __getattr__(self, item):
    """Wrap the dot access ('self.'). """
    channels = super(CondNeuGroup, self).__getattribute__('channels')
    if item in channels:
      return channels[item]
    else:
      return super(CondNeuGroup, self).__getattribute__(item)


class Channel(brainobjects.Channel):
  """Base class to model ion channels.

  Notes
  -----

  The ``__init__()`` function in :py:class:`Channel` is used to specify
  the parameters of the channel. The ``__call__()`` function
  is used to initialize the variables in this channel.
  """
  target_backend = 'jax'

  def __init__(self, host, method, **kwargs):
    super(Channel, self).__init__(**kwargs)

    self.host = host
    if not isinstance(host, CondNeuGroup):
      raise bp.errors.BrainPyError(f'Only support host with instance of {str(CondNeuGroup)}, while we got {host}')
    self.integral = bp.odeint(self.derivative, method=method)

  def update(self, _t, _dt, **kwargs):
    """The function to specify the updating rule."""
    raise NotImplementedError(f'Subclass must implement "update" function.')

  def derivative(self, *args, **kwargs):
    raise NotImplementedError

  @classmethod
  def make(cls, **params):
    """Set the default parameters for later class initialization.
    Return a tuple of `(cls, params)`. """
    raise NotImplementedError


class MolChannel(Channel):
  pass


class IonChannel(Channel):
  def current(self, *args, **kwargs):
    raise NotImplementedError
