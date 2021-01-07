import brainpy as bp
import numpy as np

def get_voltage_jump(post_has_refractory=False, mode='scalar'):
    """Voltage jump synapses without post-synaptic neuron refractory.

    .. math::

        I_{syn} = \sum J \delta(t-t_j)


    ST refers to synapse state, members of ST are listed below:
    
    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    g               0.                Synapse conductance on post-synaptic neuron.
    =============== ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        post_has_refractory (bool): whether the post-synaptic neuron have refractory.

    Returns:
        bp.SynType.
    
  
    """
    

    requires = dict(
        pre=bp.types.NeuState(['spike'])
    )

    if post_has_refractory:
        requires['post'] = bp.types.NeuState(['V', 'refractory'])

        @bp.delayed
        def output(ST, post):
            post['V'] += ST['s'] * (1. - post['refractory'])

    else:
        requires['post'] = bp.types.NeuState(['V'])

        @bp.delayed
        def output(ST, post):
            post['V'] += ST['s']

    if mode=='vector':
        requires['pre2post']=bp.types.ListConn()

        def update(ST, pre, post, pre2post):
            num_post = post['V'].shape[0]
            s = np.zeros_like(num_post, dtype=np.float_)
            spike_idx = np.where(pre['spike'] > 0.)[0]
            for i in spike_idx:
                post_ids = pre2post[i]
                s[post_ids] = 1.
            ST['s'] = s

    elif mode=='scalar':

        def update(ST, pre):
            ST['s'] = 0.
            if pre['spike'] > 0.:
                ST['s'] = 1.        

    elif mode=='matrix':
        def update(ST, pre):
            ST['s'] += pre['spike'].reshape((-1, 1))

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='voltage_jump_synapse',
                      ST=bp.types.SynState(['s']), 
                      requires=requires,
                      steps=(update, output),
                      mode = mode)
