# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
from .base import Synapse

__all__ = [
  'STP'
]


class STP(Synapse):
  r"""Short-term plasticity model.

  **Model Descriptions**

  Short-term plasticity (STP) [1]_ [2]_ [3]_, also called dynamical synapses,
  refers to a phenomenon in which synaptic efficacy changes over time in a way
  that reflects the history of presynaptic activity. Two types of STP, with
  opposite effects on synaptic efficacy, have been observed in experiments.
  They are known as Short-Term Depression (STD) and Short-Term Facilitation (STF).

  In the model proposed by Tsodyks and Markram [4]_ [5]_, the STD effect is
  modeled by a normalized variable :math:`x (0 \le x \le 1)`, denoting the fraction
  of resources that remain available after neurotransmitter depletion.
  The STF effect is modeled by a utilization parameter :math:`u`, representing
  the fraction of available resources ready for use (release probability).
  Following a spike,

  - (i) :math:`u` increases due to spike-induced calcium influx to the presynaptic
    terminal, after which
  - (ii) a fraction :math:`u` of available resources is consumed to produce the
    post-synaptic current.

  Between spikes, :math:`u` decays back to zero with time constant :math:`\tau_f`
  and :math:`x` recovers to 1 with time constant :math:`\tau_d`.

  In summary, the dynamics of STP is given by

  .. math::

      \begin{aligned}
      \frac{du}{dt} & =  -\frac{u}{\tau_f}+U(1-u^-)\delta(t-t_{sp}),\nonumber \\
      \frac{dx}{dt} & =  \frac{1-x}{\tau_d}-u^+x^-\delta(t-t_{sp}), \\
      \frac{dI}{dt} & =  -\frac{I}{\tau_s} + Au^+x^-\delta(t-t_{sp}),
      \end{aligned}

  where :math:`t_{sp}` denotes the spike time and :math:`U` is the increment
  of :math:`u` produced by a spike. :math:`u^-, x^-` are the corresponding
  variables just before the arrival of the spike, and :math:`u^+`
  refers to the moment just after the spike. The synaptic current generated
  at the synapse by the spike arriving at :math:`t_{sp}` is then given by

  .. math::

      \Delta I(t_{spike}) = Au^+x^-

  where :math:`A` denotes the response amplitude that would be produced
  by total release of all the neurotransmitter (:math:`u=x=1`), called
  absolute synaptic efficacy of the connections.

  **Model Examples**

  - `Simple illustrated example <../synapses/STP.ipynb>`_
  - `STP for Working Memory Capacity <../../examples/working_memory/Mi_2017_working_memory_capacity.ipynb>`_


  **Model Parameters**

  ============= ============== ======== ===========================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -------------------------------------------
  tau_d         200            ms       Time constant of short-term depression.
  tau_f         1500           ms       Time constant of short-term facilitation.
  U             .15            \        The increment of :math:`u` produced by a spike.
  A             1              \        The response amplitude that would be produced by total release of all the neurotransmitter
  delay         0              ms       The decay time of the current :math:`I` output onto the post-synaptic neuron groups.
  ============= ============== ======== ===========================================


  **Model Variables**

  =============== ================== =====================================================================
  **Member name** **Initial values** **Explanation**
  --------------- ------------------ ---------------------------------------------------------------------
  u                 0                 Release probability of the neurotransmitters.
  x                 1                 A Normalized variable denoting the fraction of remain neurotransmitters.
  I                 0                 Synapse current output onto the post-synaptic neurons.
  =============== ================== =====================================================================

  **References**

  .. [1] Stevens, Charles F., and Yanyan Wang. "Facilitation and depression
         at single central synapses." Neuron 14, no. 4 (1995): 795-802.
  .. [2] Abbott, Larry F., J. A. Varela, Kamal Sen, and S. B. Nelson. "Synaptic
         depression and cortical gain control." Science 275, no. 5297 (1997): 221-224.
  .. [3] Abbott, L. F., and Wade G. Regehr. "Synaptic computation."
         Nature 431, no. 7010 (2004): 796-803.
  .. [4] Tsodyks, Misha, Klaus Pawelzik, and Henry Markram. "Neural networks
         with dynamic synapses." Neural computation 10.4 (1998): 821-835.
  .. [5] Tsodyks, Misha, and Si Wu. "Short-term synaptic plasticity."
         Scholarpedia 8, no. 10 (2013): 3153.

  """

  def __init__(self, pre, post, conn, U=0.15, tau_f=1500., tau_d=200.,
               tau=8., A=1., delay=0., method='exponential_euler', **kwargs):
    super(STP, self).__init__(pre=pre, post=post, conn=conn, method=method, **kwargs)

    # parameters
    self.tau_d = tau_d
    self.tau_f = tau_f
    self.tau = tau
    self.U = U
    self.A = A
    self.delay = delay

    # variables
    self.x = bm.Variable(bm.ones(self.num, dtype=bm.float_))
    self.u = bm.Variable(bm.zeros(self.num, dtype=bm.float_))
    self.I = bm.Variable(bm.zeros(self.num, dtype=bm.float_))
    self.delayed_I = self.register_constant_delay('dI', self.num, delay=delay)

  def derivative(self, I, u, x, t):
    dIdt = -I / self.tau
    dudt = - u / self.tau_f
    dxdt = (1 - x) / self.tau_d
    return dIdt, dudt, dxdt

  def numpy_update(self, _t, _dt):
    delayed_I = self.delayed_I.pull()
    self.I[:], u, x = self.integral(self.I, self.u, self.x, _t, dt=_dt)
    for i in range(self.num):
      pre_id, post_id = self.pre_ids[i], self.post_ids[i]
      if self.pre.spike[pre_id]:
        u[i] += self.U * (1 - self.u[i])
        x[i] -= u[i] * self.x[i]
        self.I[i] += self.A * u[i] * self.x[i]
      self.post.input[post_id] += delayed_I[i]
    self.u[:] = u
    self.x[:] = x
    self.delayed_I.push(self.I)

  def jax_update(self, _t, _dt):
    pass

