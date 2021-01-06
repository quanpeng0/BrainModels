import brainpy as bp
import numpy as np
import sys

def get_Oja(gamma = 0.005, w_max = 1., w_min = 0., mode = 'vector'):
    """
    Oja's learning rule.

    .. math::
        
        \\frac{d w_{ij}}{dt} = \\gamma(\\upsilon_i \\upsilon_j - w_{ij}\\upsilon_i ^ 2)
        
        
    **Learning Rule Parameters**
    
    ============= ============== ======== ================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------
    gamma         0.005          \        Learning rate.

    w_max         1.             \        Maximal possible synapse weight.

    w_min         0.             \        Minimal possible synapse weight.

    mode          'vector'       \        Data structure of ST members.
    ============= ============== ======== ================================

    Returns:
        bp.Syntype: return description of synapse with Oja's rule.
        
    
    **Learning Rule State**
    
    ST refers to synapse state (note that Oja learning rule can be implemented as synapses),
    members of ST are listed below:
    
    ================ ================= =========================================================
    **Member name**  **Initial Value** **Explanation**
    ---------------- ----------------- ---------------------------------------------------------
    w                0.05              Synapse weight.

    output_save      0.                Temporary save synapse output value until post-synaptic
                                       neuron get the value after delay time.
    ================ ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).        
        
    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """
    
    
    ST = bp.types.SynState({'w': 0.05, 'output_save': 0.})
    
    requires = dict(
        pre = bp.types.NeuState(['r']),
        post = bp.types.NeuState(['r']), 
        post2syn=bp.types.ListConn(),
        post2pre=bp.types.ListConn(),
    )
    
    @bp.integrate
    def int_w(w, t, r_pre, r_post):
        dw = gamma * (r_post * r_pre - r_post * r_post * w)
        return dw

    def update(ST, _t, pre, post, post2pre, post2syn):
        for i in range(len(post2pre)):
            pre_ids = post2pre[i]
            syn_ids = post2syn[i]
            post['r'] = np.sum(ST['w'][syn_ids] * pre['r'][pre_ids])
            ST['w'][syn_ids] = int_w(ST['w'][syn_ids], _t,  pre['r'][pre_ids], post['r'][i])
    
    if mode == 'scalar':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    elif mode == 'vector':
        return bp.SynType(name='Oja_synapse',
                          ST=ST,
                          requires=requires,
                          steps=update,
                          mode=mode)
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
