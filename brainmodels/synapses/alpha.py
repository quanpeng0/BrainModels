# -*- coding: utf-8 -*-

from .dual_exp import *

__all__ = [
  'AlphaCUBA', 'AlphaCOBA'
]


class AlphaCUBA(DualExpCUBA):
  r"""Current-based alpha synapse model.

  **Model Descriptions**

  The analytical expression of alpha synapse is given by:

  .. math::

      g_{syn}(t)= g_{max} \frac{t-t_{s}}{\tau} \exp \left(-\frac{t-t_{s}}{\tau}\right).

  While, this equation is hard to implement. So, let's try to convert it into the
  differential forms:

  .. math::

      \begin{aligned}
      &g_{\mathrm{syn}}(t)= g_{\mathrm{max}} g \\
      &\frac{d g}{d t}=-\frac{g}{\tau}+h \\
      &\frac{d h}{d t}=-\frac{h}{\tau}+\delta\left(t_{0}-t\right)
      \end{aligned}

  The current onto the post-synaptic neuron is given by

  .. math::

      I_{syn}(t) = g_{\mathrm{syn}}(t).


  **Model Examples**

  - `Simple illustrated example <../synapses/alphacuba.ipynb>`_


  **Model Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  tau_decay     2              ms       The decay time constant of the synaptic state.
  g_max         .2             µmho(µS) The maximum conductance.
  ============= ============== ======== ===================================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  g                   0                Synapse conductance on the post-synaptic neuron.
  h                   0                Gating variable.
  pre_spike           False            The history spiking states of the pre-synaptic neurons.
  ================== ================= =========================================================

  **References**

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.
  """

  def __init__(self, pre, post, conn, delay=0., g_max=1., tau_decay=10.0,
               update_type='loop', **kwargs):
    super(AlphaCUBA, self).__init__(pre=pre, post=post, conn=conn,
                                    delay=delay,
                                    g_max=g_max,
                                    tau_decay=tau_decay,
                                    tau_rise=tau_decay,
                                    update_type=update_type,
                                    **kwargs)


class AlphaCOBA(DualExpCOBA):
  """Conductance-based alpha synapse model.

  **Model Descriptions**

  The conductance-based alpha synapse model is similar with the
  `current-based alpha synapse model <./brainmodels.synapses.AlphaCUBA.rst>`_,
  except the expression which output onto the post-synaptic neurons:

  .. math::

      I_{syn}(t) = g_{\mathrm{syn}}(t) (V(t)-E)

  where :math:`V(t)` is the membrane potential of the post-synaptic neuron,
  :math:`E` is the reversal potential.


  **Model Examples**

  - `Simple illustrated example <../synapses/alphacoba.ipynb>`_


  **Model Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  tau_decay     2              ms       The decay time constant of the synaptic state.
  g_max         .2             µmho(µS) The maximum conductance.
  E             0              mV       The reversal potential for the synaptic current.
  ============= ============== ======== ===================================================================================


  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  g                   0                Synapse conductance on the post-synaptic neuron.
  h                   0                Gating variable.
  pre_spike           False            The history spiking states of the pre-synaptic neurons.
  ================== ================= =========================================================

  **References**

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.

  """

  def __init__(self, pre, post, conn, delay=0., g_max=1., tau_decay=10.0,
               E=0., update_type='loop', **kwargs):
    super(AlphaCOBA, self).__init__(pre=pre, post=post, conn=conn,
                                    delay=delay,
                                    g_max=g_max,
                                    E=E,
                                    tau_decay=tau_decay,
                                    tau_rise=tau_decay,
                                    update_type=update_type,
                                    **kwargs)
