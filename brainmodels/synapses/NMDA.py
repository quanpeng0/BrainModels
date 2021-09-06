# -*- coding: utf-8 -*-

import brainpy as bp
from numba import prange

__all__ = [
  'NMDA'
]


class NMDA(bp.TwoEndConn):
  """NMDA conductance-based synapse.

  .. math::

      & I_{syn} = \\bar{g} s (V-E_{syn}) \\cdot g_{\\infty}

      & g_{\\infty}(V,[{Mg}^{2+}]_{o}) = (1+{e}^{-\\alpha V}
      \\frac{[{Mg}^{2+}]_{o}} {\\beta})^{-1} 

      & \\frac{d s_{j}(t)}{dt} = -\\frac{s_{j}(t)}
      {\\tau_{decay}}+a x_{j}(t)(1-s_{j}(t)) 

      & \\frac{d x_{j}(t)}{dt} = -\\frac{x_{j}(t)}{\\tau_{rise}}+
      \\sum_{k} \\delta(t-t_{j}^{k})


  where the decay time of NMDA currents is taken to be :math:`\\tau_{decay}` =100 ms,
  :math:`a= 0.5 ms^{-1}`, and :math:`\\tau_{rise}` =2 ms


  **Synapse Parameters**

  ============= ============== =============== ================================================
  **Parameter** **Init Value** **Unit**        **Explanation**
  ------------- -------------- --------------- ------------------------------------------------
  g_max         .15            µmho(µS)        Maximum conductance.

  E             0.             mV              The reversal potential for the synaptic current.

  alpha         .062           \               Binding constant.

  beta          3.57           \               Unbinding constant.

  cc_Mg         1.2            mM              Concentration of Magnesium ion.

  tau_decay     100.           ms              The time constant of decay.

  tau_rise      2.             ms              The time constant of rise.

  a             .5             1/ms 

  mode          'scalar'       \               Data structure of ST members.
  ============= ============== =============== ================================================    
  
  **Synapse State**

  ST refers to the synapse state, items in ST are listed below:
  
  =============== ================== =========================================================
  **Member name** **Initial values** **Explanation**
  --------------- ------------------ --------------------------------------------------------- 
  s               0                     Gating variable.
  
  g               0                     Synapse conductance.

  x               0                     Gating variable.
  =============== ================== =========================================================
      
  References
  ----------
  
  .. [1] Brunel N, Wang X J. Effects of neuromodulation in a 
         cortical network model of object working memory dominated 
         by recurrent inhibition[J]. 
         Journal of computational neuroscience, 2001, 11(1): 63-85.
  
  """

  def __init__(self, pre, post, conn, delay=0., g_max=0.15, E=0., cc_Mg=1.2,
               alpha=0.062, beta=3.57, tau_decay=100., a=0.5, tau_rise=2.,
               update_type='loop', **kwargs):
    super(NMDA, self).__init__(pre=pre, post=post, **kwargs)

    assert hasattr(pre, 'spike'), 'Pre-synaptic group must has "spike" variable.'
    assert hasattr(post, 'V'), 'Post-synaptic group must has "V" variable.'
    assert hasattr(post, 'input'), 'Post-synaptic group must has "input" variable.'

    # parameters
    self.g_max = g_max
    self.E = E
    self.alpha = alpha
    self.beta = beta
    self.cc_Mg = cc_Mg
    self.tau_decay = tau_decay
    self.tau_rise = tau_rise
    self.a = a
    self.delay = delay

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
    self.s = bp.math.Variable(bp.math.zeros(self.size))
    self.x = bp.math.Variable(bp.math.zeros(self.size))
    self.g = self.register_constant_delay('g', size=self.size, delay=delay)

  @bp.odeint(method='exponential_euler')
  def integral(self, s, x, t):
    dxdt = -x / self.tau_rise
    dsdt = -s / self.tau_decay + self.a * x * (1 - s)
    return dsdt, dxdt

  def _loop_update(self, _t, _dt):
    delayed_g = self.g.pull()
    self.s[:], self.x[:] = self.integral(self.s, self.x, _t, dt=_dt)
    for i in prange(self.size):
      pre_id, post_id = self.pre_ids[i], self.post_ids[i]
      self.x[i] += self.pre.spike[pre_id]
      g_inf = 1 + self.cc_Mg / self.beta * bp.math.exp(-self.alpha * self.post.V[post_id])
      self.post.input[post_id] -= delayed_g[i] * (self.post.V[post_id] - self.E) / g_inf
    self.g.push(self.g_max * self.s)
