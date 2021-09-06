# -*- coding: utf-8 -*-

from .dual_exp import *

__all__ = [
  'AlphaCUBA', 'AlphaCOBA'
]


class AlphaCUBA(DualExpCUBA):
  r"""Alpha synapse model.

  .. math::

      g_{syn}(t)=\bar{g}_{syn} \frac{t-t_{s}}{\tau} \exp \left(-\frac{t-t_{s}}{\tau}\right)

  This equation can be rewritten as the differential forms:

  .. math::

      \begin{aligned}
      &g_{\mathrm{syn}}(t)=\bar{g}_{\mathrm{syn}} g \\
      &\frac{d g}{d t}=-\frac{g}{\tau}+h \\
      &\frac{d h}{d t}=-\frac{h}{\tau}+\delta\left(t_{0}-t\right)
      \end{aligned}

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
  tau_decay     2.             ms       The time constant of decay.

  g_max         .2             µmho(µS) Maximum conductance.

  E             0.             mV       The reversal potential for the synaptic current. (only for conductance-based model)

  co_base       False          \        Whether to return Conductance-based model. If False: return current-based model.
  ============= ============== ======== ===================================================================================

  **Synapse Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  g                  0                  Synapse conductance on the post-synaptic neuron.
  s                  0                  Gating variable.
  x                  0                  Gating variable.
  ================ ================== =========================================================

  References
  ----------
  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.
  """

  def __init__(self, pre, post, conn, delay=0., g_max=1., tau_decay=10.0,
               update_type='loop', **kwargs):
    super(AlphaCUBA, self).__init__(pre=pre, post=post, conn=conn,
                                    delay=delay, g_max=g_max,
                                    tau_decay=tau_decay, tau_rise=tau_decay,
                                    update_type=update_type, **kwargs)


class AlphaCOBA(DualExpCOBA):
  def __init__(self, pre, post, conn, delay=0., g_max=1., tau_decay=10.0,
               E=0., update_type='loop', **kwargs):
    super(AlphaCOBA, self).__init__(pre=pre, post=post, conn=conn,
                                    delay=delay, g_max=g_max, E=E,
                                    tau_decay=tau_decay, tau_rise=tau_decay,
                                    update_type=update_type, **kwargs)
