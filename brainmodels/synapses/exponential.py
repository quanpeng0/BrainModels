# -*- coding: utf-8 -*-


import brainpy.math as bm

from .base import Synapse

__all__ = [
  'ExpCUBA', 'ExpCOBA'
]


class ExpCUBA(Synapse):
  r"""Current-based exponential decay synapse model.

  **Model Descriptions**

  The single exponential decay synapse model assumes the release of neurotransmitter,
  its diffusion across the cleft, the receptor binding, and channel opening all happen
  very quickly, so that the channels instantaneously jump from the closed to the open state.
  Therefore, its expression is given by

  .. math::

      g_{\mathrm{syn}}(t)=g_{\mathrm{max}} e^{-\left(t-t_{0}\right) / \tau}

  where :math:`\tau_{delay}` is the time constant of the synaptic state decay,
  :math:`t_0` is the time of the pre-synaptic spike,
  :math:`g_{\mathrm{max}}` is the maximal conductance.

  Accordingly, the differential form of the exponential synapse is given by

  .. math::

      \begin{aligned}
       & g_{\mathrm{syn}}(t) = g_{max} g \\
       & \frac{d g}{d t} = -\frac{g}{\tau_{decay}}+\sum_{k} \delta(t-t_{j}^{k}).
       \end{aligned}

  For the current output onto the post-synaptic neuron, its expression is given by

  .. math::

      I_{\mathrm{syn}}(t) = g_{\mathrm{syn}}(t)


  **Model Examples**

  - `Simple illustrated example <../synapses/exp_cuba.ipynb>`_
  - `(Brunel & Hakim, 1999) Fast Global Oscillation <../../examples/oscillation_synchronization/Brunel_Hakim_1999_fast_oscillation.ipynb>`_
  - `(Vreeswijk & Sompolinsky, 1996) E/I balanced network <../../examples/ei_nets/Vreeswijk_1996_EI_net.ipynb>`_
  - `(Brette, et, al., 2007) CUBA <../../examples/ei_nets/Brette_2007_CUBA.ipynb>`_
  - `(Tian, et al., 2020) E/I Net for fast response <../../examples/ei_nets/Tian_2020_EI_net_for_fast_response.ipynb>`_


  **Model Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  tau_decay     8              ms       The time constant of decay.
  g_max         1              µmho(µS) The maximum conductance.
  ============= ============== ======== ===================================================================================

  **Model Variables**

  ================ ================== =========================================================
  **Member name**  **Initial values** **Explanation**
  ---------------- ------------------ ---------------------------------------------------------
  g                 0                 Gating variable.
  pre_spike         False             The history spiking states of the pre-synaptic neurons.
  ================ ================== =========================================================

  **References**

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.
  """

  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0,
               method='exponential_euler', **kwargs):
    super(ExpCUBA, self).__init__(pre=pre, post=post, conn=conn, method=method,
                                  build_conn=False, **kwargs)

    # parameters
    self.tau = tau
    self.delay = delay
    self.g_max = g_max

    # connection
    self.num = self.post.num
    if bm.is_numpy_backend():
      self.pre2post = self.conn.require('pre2post')
      self.target_backend = 'numpy'
    else:
      self.post2pre_mat = self.conn.require('post2pre_mat')
      self.target_backend = 'jax'

    # variables
    self.g = bm.Variable(bm.zeros(self.post.num))
    self.pre_spike = self.register_constant_delay('pre_spike', self.pre.num, delay)

  def derivative(self, g, t):
    dg = -g / self.tau
    return dg

  def numpy_update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    pre_spike = self.pre_spike.pull()
    self.g[:] = self.integral(self.g, _t, dt=_dt)
    for pre_id in range(self.pre.num):
      if pre_spike[pre_id]:
        post_ids = self.pre2post[pre_id]
        self.g[post_ids] += self.g_max
    self.post.input += self.g

  def jax_update(self, _t, _dt):
      self.pre_spike.push(self.pre.spike)
      delayed_pre_spike = self.pre_spike.pull()
      post_sp = bm.pre2post(delayed_pre_spike, self.post2pre_mat)
      self.g.value = self.integral(self.g.value, _t, dt=_dt) + post_sp * self.g_max
      self.post.input += self.g


class ExpCOBA(ExpCUBA):
  """Conductance-based exponential decay synapse model.

  **Model Descriptions**

  The conductance-based exponential decay synapse model is similar with the
  `current-based exponential decay synapse model <./brainmodels.synapses.ExpCUBA.rst>`_,
  except the expression which output onto the post-synaptic neurons:

  .. math::

      I_{syn}(t) = g_{\mathrm{syn}}(t) (V(t)-E)

  where :math:`V(t)` is the membrane potential of the post-synaptic neuron,
  :math:`E` is the reversal potential.


  **Model Examples**

  - `Simple illustrated example <../synapses/exp_coba.ipynb>`_
  - `(Brette, et, al., 2007) COBA <../../examples/ei_nets/Brette_2007_COBA.ipynb>`_
  - `(Brette, et, al., 2007) COBAHH <../../examples/ei_nets/Brette_2007_COBAHH.ipynb>`_


  **Model Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  tau_decay     8              ms       The time constant of decay.
  g_max         1              µmho(µS) The maximum conductance.
  E             0              mV       The reversal potential for the synaptic current.
  ============= ============== ======== ===================================================================================

  **Model Variables**

  ================ ================== =========================================================
  **Member name**  **Initial values** **Explanation**
  ---------------- ------------------ ---------------------------------------------------------
  g                 0                 Gating variable.
  pre_spike         False             The history spiking states of the pre-synaptic neurons.
  ================ ================== =========================================================

  **References**

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.
  """

  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0, E=0.,
               method='exponential_euler', **kwargs):
    super(ExpCOBA, self).__init__(pre=pre, post=post, conn=conn,
                                  g_max=g_max, delay=delay, tau=tau,
                                  method=method, **kwargs)

    self.E = E

  def numpy_update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    pre_spike = self.pre_spike.pull()
    self.g[:] = self.integral(self.g, _t, dt=_dt)
    for pre_id in range(self.pre.num):
      if pre_spike[pre_id]:
        post_ids = self.pre2post[pre_id]
        self.g[post_ids] += self.g_max
    self.post.input += self.g * (self.E - self.post.V)

  def jax_update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    delayed_pre_spike = self.pre_spike.pull()
    post_sp = bm.pre2post(delayed_pre_spike, self.post2pre_mat)
    self.g.value = self.integral(self.g.value, _t, dt=_dt) + post_sp * self.g_max
    self.post.input += self.g * (self.E - self.post.V)
