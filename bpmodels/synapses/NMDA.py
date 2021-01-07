# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np

def get_NMDA(g_max=0.15, E=0, alpha=0.062, beta=3.57, 
            cc_Mg=1.2, tau_decay=100., a=0.5, tau_rise=2., mode = 'scalar'):
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
    
    
    Returns:
        bp.Syntype: return description of the NMDA synapse model.

    **Synapse State**

    ST refers to the synapse state, items in ST are listed below:
    
    =============== ================== =========================================================
    **Member name** **Initial values** **Explanation**
    --------------- ------------------ --------------------------------------------------------- 
    s               0                     Gating variable.
    
    g               0                     Synapse conductance.

    x               0                     Gating variable.
    =============== ================== =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).
        
    References:
        .. [1] Brunel N, Wang X J. Effects of neuromodulation in a 
               cortical network model of object working memory dominated 
               by recurrent inhibition[J]. 
               Journal of computational neuroscience, 2001, 11(1): 63-85.
    
    """

    @bp.integrate
    def int_x(x, t):
        return -x / tau_rise

    @bp.integrate
    def int_s(s, t, x):
        return -s / tau_decay + a * x * (1 - s)

    ST=bp.types.SynState({'s': 0., 'x': 0., 'g': 0.})

    requires = dict(
        pre=bp.types.NeuState(['spike']),
        post=bp.types.NeuState(['V', 'input'])
    )

    if mode == 'scalar':
        def update(ST, _t, pre):
            x = int_x(ST['x'], _t)
            x += pre['spike']
            s = int_s(ST['s'], _t, x)
            ST['x'] = x
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post):
            g_inf = 1 + cc_Mg / beta * np.exp(-alpha * post['V'])
            post['input'] -= ST['g'] * (post['V'] - E) / g_inf

    elif mode == 'vector':
        requires['pre2syn']=bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index')
        requires['post_slice_syn']=bp.types.Array(dim=2)

        def update(ST, _t, pre, pre2syn):
            for pre_id in range(len(pre2syn)):
                if pre['spike'][pre_id] > 0.:
                    syn_ids = pre2syn[pre_id]
                    ST['x'][syn_ids] += 1.
            x = int_x(ST['x'], _t)
            s = int_s(ST['s'], _t, x)
            ST['x'] = x
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post, post_slice_syn):
            g_inf = 1 + cc_Mg / beta * np.exp(-alpha * post['V'])
            
            num_post = post_slice_syn.shape[0]
            g = np.zeros(num_post, dtype=np.float_)
            for post_id in range(num_post):
                pos = post_slice_syn[post_id]
                g[post_id] = np.sum(ST['g'][pos[0]: pos[1]])  
            post['input'] -= g * (post['V'] - E) / g_inf

    elif mode == 'matrix':
        requires['conn_mat']=bp.types.MatConn()

        def update(ST, _t, pre, conn_mat):
            x = int_x(ST['x'], _t)
            x += pre['spike'].reshape((-1, 1)) * conn_mat
            s = int_s(ST['s'], _t, x)
            ST['x'] = x
            ST['s'] = s
            ST['g'] = g_max * s

        @bp.delayed
        def output(ST, post):
            g = np.sum(ST['g'], axis=0)
            g_inf = 1 + cc_Mg / beta * np.exp(-alpha * post['V'])
            post['input'] -= g * (post['V'] - E) / g_inf

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))


    return bp.SynType(name='NMDA_synapse',
                      ST=ST, requires=requires,
                      steps=(update, output),
                      mode = mode)

