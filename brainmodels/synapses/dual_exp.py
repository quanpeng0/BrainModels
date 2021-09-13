# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

__all__ = [
  'DualExpCUBA', 'DualExpCOBA',
]


class DualExpCUBA(bp.TwoEndConn):
  r"""Dual exponential synapse model.

  .. math::

      &\frac {ds} {dt} = x
      
      \tau_{1} \tau_{2} \frac {dx}{dt} = - & (\tau_{1}+\tau_{2})x 
      -s + \sum \delta(t-t^f)

  For conductance-based (co-base=True):

  .. math::
  
      I_{syn}(t) = g_{syn} (t) (V(t)-E_{syn})

  For current-based (co-base=False):

  .. math::
  
      I(t) = \bar{g} s (t)


  **Synapse Parameters**
  
  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  tau_1         1.             ms       Time constant.

  tau_2         3.             ms       Time constant.

  g_max         .2             µmho(µS) Maximum conductance.

  E             0.             mV       The reversal potential for the synaptic current. (only for conductance-based model)

  co_base       False          \        Whether to return Conductance-based model. If False: return current-based model.

  mode          'scalar'       \        Data structure of ST members.
  ============= ============== ======== ===================================================================================  

  **Synapse State**
      
  ST refers to synapse state, members of ST are listed below:

  ================ ================== =========================================================
  **Member name**  **Initial values** **Explanation**
  ---------------- ------------------ ---------------------------------------------------------    
  g                  0                  Synapse conductance on the post-synaptic neuron.

  s                  0                  Gating variable.

  x                  0                  Gating variable.                              
  ================ ================== =========================================================

  References:
  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
         "The Synapse." Principles of Computational Modelling in Neuroscience.
         Cambridge: Cambridge UP, 2011. 172-95. Print.
  """

  def __init__(self, pre, post, conn, delay=0., g_max=1., tau_decay=10.0,
               tau_rise=1., update_type='loop', **kwargs):
    super(DualExpCUBA, self).__init__(pre=pre, post=post, conn=conn, **kwargs)

    # checking
    assert hasattr(pre, 'spike'), 'Pre-synaptic group must has "spike" variable.'
    assert hasattr(post, 'input'), 'Post-synaptic group must has "input" variable.'

    # parameters
    self.tau_rise = tau_rise
    self.tau_decay = tau_decay
    self.delay = delay
    self.g_max = g_max

    # connections
    if update_type == 'loop':
      self.pre_ids, self.post_ids = self.conn.requires('pre_ids', 'post_ids')
      self.update = self._loop_update
      self.size = len(self.pre_ids)
      self.target_backend = 'numpy'

    elif update_type == 'loop_slice':
      raise NotImplementedError

    elif update_type == 'matrix':
      raise NotImplementedError

    else:
      raise bp.errors.UnsupportedError(f'Do not support {update_type} method.')

    # variables
    self.g = bm.Variable(bm.zeros(self.size))
    self.h = bm.Variable(bm.zeros(self.size))
    self.pre_spike = self.register_constant_delay('pre_spike', size=self.size, delay=delay)

  @bp.odeint(method='exponential_euler')
  def integral(self, g, h, t):
    dgdt = -g / self.tau_decay + h
    dhdt = -h / self.tau_rise
    return dgdt, dhdt

  def _loop_update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    pre_spike = self.pre_spike.pull()

    self.g[:], self.h[:] = self.integral(self.g[:], self.h[:], _t, dt=_dt)
    for i in range(self.size):
      pre_id, post_id = self.pre_ids[i], self.post_ids[i]
      self.h[i] += pre_spike[pre_id]
      self.post.input[post_id] += self.g_max * self.g[i]


class DualExpCOBA(DualExpCUBA):
  def __init__(self, pre, post, conn, delay=0., g_max=1., tau_decay=10.0,
               tau_rise=1., E=0., update_type='loop', **kwargs):
    super(DualExpCOBA, self).__init__(pre, post, conn, delay=delay, g_max=g_max,
                                      tau_decay=tau_decay, tau_rise=tau_rise,
                                      update_type=update_type, **kwargs)

    self.E = E
    assert hasattr(post, 'V'), 'Post-synaptic group must has "V" variable.'

  def _loop_update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    pre_spike = self.pre_spike.pull()

    self.g[:], self.h[:] = self.integral(self.g[:], self.h[:], _t, dt=_dt)
    for i in range(self.size):
      pre_id, post_id = self.pre_ids[i], self.post_ids[i]
      self.h[i] += pre_spike[pre_id]
      self.post.input[post_id] += self.g_max * self.g[i] * (self.E - self.post.V[post_id])
