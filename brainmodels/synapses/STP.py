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

  - `STP for Working Memory Capacity <https://brainpy-examples.readthedocs.io/en/latest/working_memory/Mi_2017_working_memory_capacity.html>`_

  **STD**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import brainmodels
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = brainmodels.neurons.LIF(1)
    >>> neu2 = brainmodels.neurons.LIF(1)
    >>> syn1 = brainmodels.synapses.STP(neu1, neu2, bp.connect.All2All(), U=0.2, tau_d=150., tau_f=2.)
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.StructRunner(net, inputs=[('pre.input', 28.)], monitors=['syn.I', 'syn.u', 'syn.x'])
    >>> runner.run(150.)
    >>>
    >>>
    >>> # plot
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 7)
    >>>
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.u'][:, 0], label='u')
    >>> plt.plot(runner.mon.ts, runner.mon['syn.x'][:, 0], label='x')
    >>> plt.legend()
    >>>
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.I'][:, 0], label='I')
    >>> plt.legend()
    >>>
    >>> plt.xlabel('Time (ms)')
    >>> plt.show()

  **STF**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import brainmodels
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = brainmodels.neurons.LIF(1)
    >>> neu2 = brainmodels.neurons.LIF(1)
    >>> syn1 = brainmodels.synapses.STP(neu1, neu2, bp.connect.All2All(), U=0.1, tau_d=10, tau_f=100.)
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.StructRunner(net, inputs=[('pre.input', 28.)], monitors=['syn.I', 'syn.u', 'syn.x'])
    >>> runner.run(150.)
    >>>
    >>>
    >>> # plot
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 7)
    >>>
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.u'][:, 0], label='u')
    >>> plt.plot(runner.mon.ts, runner.mon['syn.x'][:, 0], label='x')
    >>> plt.legend()
    >>>
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.I'][:, 0], label='I')
    >>> plt.legend()
    >>>
    >>> plt.xlabel('Time (ms)')
    >>> plt.show()



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
               tau=8., A=1., delay=0., method='exp_auto', name=None):
    super(STP, self).__init__(pre=pre, post=post, conn=conn, method=method, name=name)
    self.check_post_attrs('input')
    self.check_pre_attrs('spike')

    # parameters
    self.tau_d = tau_d
    self.tau_f = tau_f
    self.tau = tau
    self.U = U
    self.A = A
    self.delay = delay

    # connections
    self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')

    # variables
    num = len(self.pre_ids)
    self.x = bm.Variable(bm.ones(num, dtype=bm.float_))
    self.u = bm.Variable(bm.zeros(num, dtype=bm.float_))
    self.I = bm.Variable(bm.zeros(num, dtype=bm.float_))
    self.delayed_I = bp.ConstantDelay(num, delay=delay)

  @property
  def derivative(self):
    dI = lambda I, t: -I / self.tau
    du = lambda u, t: - u / self.tau_f
    dx = lambda x, t: (1 - x) / self.tau_d
    return bp.JointEq([dI, du, dx])

  def update(self, _t, _dt):
    delayed_I = self.delayed_I.pull()
    self.post.input += bm.syn2post(delayed_I, self.post_ids, self.post.num)
    self.I.value, u, x = self.integral(self.I, self.u, self.x, _t, dt=_dt)
    syn_sps = bm.pre2syn(self.pre.spike, self.pre_ids)
    u = bm.where(syn_sps, u + self.U * (1 - self.u), u)
    x = bm.where(syn_sps, x - u * self.x, x)
    self.I.value = bm.where(syn_sps, self.I, self.I + self.A * u * self.x)
    self.u.value = u
    self.x.value = x
    self.delayed_I.push(self.I)
