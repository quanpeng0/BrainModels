# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

__all__ = [
  'DualExpCUBA', 'DualExpCOBA',
]


class DualExpCUBA(bp.TwoEndConn):
  r"""Current-based dual exponential synapse model.

  **Model Descriptions**

  The dual exponential synapse model [1]_, also named as *difference of two exponentials* model,
  is given by:

  .. math::

    g_{\mathrm{syn}}(t)=g_{\mathrm{max}} \frac{\tau_{1} \tau_{2}}{
        \tau_{1}-\tau_{2}}\left(\exp \left(-\frac{t-t_{0}}{\tau_{1}}\right)
        -\exp \left(-\frac{t-t_{0}}{\tau_{2}}\right)\right)

  where :math:`\tau_1` is the time constant of the decay phase, :math:`\tau_2`
  is the time constant of the rise phase, :math:`t_0` is the time of the pre-synaptic
  spike, :math:`g_{\mathrm{max}}` is the maximal conductance.

  However, in practice, this formula is hard to implement. The equivalent solution is
  two coupled linear differential equations [2]_:

  .. math::

      \begin{aligned}
      &g_{\mathrm{syn}}(t)=g_{\mathrm{max}} g \\
      &\frac{d g}{d t}=-\frac{g}{\tau_{\mathrm{decay}}}+h \\
      &\frac{d h}{d t}=-\frac{h}{\tau_{\text {rise }}}+ \delta\left(t_{0}-t\right),
      \end{aligned}

  The current onto the post-synaptic neuron is given by

  .. math::
  
      I_{syn}(t) = g_{\mathrm{syn}}(t).


  **Model Examples**

  - `Simple illustrated example <../synapses/dual_exp_cuba.ipynb>`_


  **Model Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  tau_decay     10             ms       The time constant of the synaptic decay phase.
  tau_rise      1              ms       The time constant of the synaptic rise phase.
  g_max         1              µmho(µS) The maximum conductance.
  ============= ============== ======== ===================================================================================


  **Model Variables**

  ================ ================== =========================================================
  **Member name**  **Initial values** **Explanation**
  ---------------- ------------------ ---------------------------------------------------------    
  g                  0                 Synapse conductance on the post-synaptic neuron.
  s                  0                 Gating variable.
  pre_spike          False             The history spiking states of the pre-synaptic neurons.
  ================ ================== =========================================================

  **References**

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
         "The Synapse." Principles of Computational Modelling in Neuroscience.
         Cambridge: Cambridge UP, 2011. 172-95. Print.
  .. [2] Roth, A., & Van Rossum, M. C. W. (2009). Modeling Synapses. Computational
         Modeling Methods for Neuroscientists.

  """

  def __init__(self, pre, post, conn, delay=0., g_max=1., tau_decay=10.0,
               tau_rise=1., update_type='loop', **kwargs):
    # initialization
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
      self.steps.replace('update', self._loop_update)
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
    self.pre_spike = self.register_constant_delay('pre_spike', size=self.pre.shape, delay=delay)

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
  """Conductance-based dual exponential synapse model.

  **Model Descriptions**

  The conductance-based dual exponential synapse model is similar with the
  `current-based dual exponential synapse model <./brainmodels.synapses.DualExpCUBA.rst>`_,
  except the expression which output onto the post-synaptic neurons:

  .. math::

      I_{syn}(t) = g_{\mathrm{syn}}(t) (V(t)-E)

  where :math:`V(t)` is the membrane potential of the post-synaptic neuron,
  :math:`E` is the reversal potential.



  **Model Examples**

  - `Simple illustrated example <../synapses/dual_exp_coba.ipynb>`_


  **Model Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  tau_decay     10             ms       The time constant of the synaptic decay phase.
  tau_rise      1              ms       The time constant of the synaptic rise phase.
  g_max         1              µmho(µS) The maximum conductance.
  E             0              mV       The reversal potential for the synaptic current.
  ============= ============== ======== ===================================================================================


  **Model Variables**

  ================ ================== =========================================================
  **Member name**  **Initial values** **Explanation**
  ---------------- ------------------ ---------------------------------------------------------
  g                  0                 Synapse conductance on the post-synaptic neuron.
  s                  0                 Gating variable.
  pre_spike          False             The history spiking states of the pre-synaptic neurons.
  ================ ================== =========================================================

  **References**

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
         "The Synapse." Principles of Computational Modelling in Neuroscience.
         Cambridge: Cambridge UP, 2011. 172-95. Print.

  """
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
