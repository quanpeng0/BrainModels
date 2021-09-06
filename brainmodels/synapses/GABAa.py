# -*- coding: utf-8 -*-

from .AMPA import AMPA

__all__ = [
  'GABAa',
]


class GABAa(AMPA):
  """
  GABAa conductance-based synapse model (markov form).

  .. math::

      I_{syn}&= - \\bar{g}_{max} s (V - E)

      \\frac{d s}{d t}&=\\alpha[T](1-s) - \\beta s

  **Synapse Parameters**

  ============= ============== ======== =======================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ---------------------------------------
  g_max         0.04           \        Maximum synapse conductance.

  E             -80.           \        Reversal potential of synapse.

  alpha         0.53           \        Activating rate constant of G protein catalyzed

                                        by activated GABAb receptor.

  beta          0.18           \        De-activating rate constant of G protein.

  T             1.             \        Transmitter concentration when synapse is

                                        triggered by a pre-synaptic spike.

  T_duration    1.             \        Transmitter concentration duration time

                                        after being triggered.
  ============= ============== ======== =======================================

  **Synapse Variables**

  An object of synapse class record those variables for each synapse:

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  s                  0.                Gating variable.

  g                  0.                Synapse conductance on post-synaptic neuron.

  t_last_pre_spike   -1e7              Last spike time stamp of pre-synaptic neuron.
  ================== ================= =========================================================

  References:
      .. [1] Destexhe, Alain, and Denis Paré. "Impact of network activity
             on the integrative properties of neocortical pyramidal neurons
             in vivo." Journal of neurophysiology 81.4 (1999): 1531-1547.
  """

  def __init__(self, pre, post, conn, delay=0., g_max=0.04, E=-80., alpha=0.53,
               beta=0.18, T=1., T_duration=1., update_type='loop', **kwargs):
    super(GABAa, self).__init__(pre, post, conn, delay=delay, g_max=g_max, E=E,
                                alpha=alpha, beta=beta, T=T, T_duration=T_duration,
                                update_type=update_type, **kwargs)
