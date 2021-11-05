# -*- coding: utf-8 -*-

import brainpy as bp
from brainmodels.neurons.cond_base import CondNeuGroup
from brainpy.simulation import brainobjects

__all__ = [
  'Channel',
]


class Channel(brainobjects.Channel):
  """Base class to model ion channels.

  Notes
  -----

  The ``__init__()`` function in :py:class:`Channel` is used to specify
  the parameters of the channel. The ``__call__()`` function
  is used to initialize the variables in this channel.
  """

  def __init__(self, **kwargs):
    super(Channel, self).__init__(**kwargs)

  def init(self, host, **kwargs):
    """Initialize variables in this channel."""
    if not isinstance(host, CondNeuGroup):
      raise bp.errors.BrainPyError(f'Only support host with instance '
                                   f'of {str(CondNeuGroup)}, while we got {host}')
    self.host = host

  def update(self, _t, _dt, **kwargs):
    """The function to specify the updating rule."""
    raise NotImplementedError(f'Subclass of {self.__class__.__name__} '
                              f'must implement "update" function.')


