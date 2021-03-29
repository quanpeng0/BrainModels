# -*- coding: utf-8 -*-
import brainpy as bp

class Two_exponentials(bp.TwoEndConn):
    '''
    two_exponentials synapse model.

    .. math::

        &\\frac {ds} {dt} = x
        
        \\tau_{1} \\tau_{2} \\frac {dx}{dt} = - & (\\tau_{1}+\\tau_{2})x 
        -s + \\sum \\delta(t-t^f)


    For conductance-based (co-base=True):

    .. math::
    
        I_{syn}(t) = g_{syn} (t) (V(t)-E_{syn})


    For current-based (co-base=False):

    .. math::
    
        I(t) = \\bar{g} s (t)


 **Synapse Parameters**
    
    ============= ============== ======== ===================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -----------------------------------------------------------------------------------
    tau_1         1.             ms       Time constant.

    tau_2         3.             ms       Time constant.

    g_max         .2             µmho(µS) Maximum conductance.

    E             0.             mV       The reversal potential for the synaptic current. (only for conductance-based model)

    co_base       False          \        Whether to return Conductance-based model. If False: return current-based model.

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ===================================================================================  
    
    Returns:
        bp.Syntype: return description of the two_exponentials synapse model.

    **Synapse State**
        
    ST refers to synapse state, members of ST are listed below:

    ================ ================== =========================================================
    **Member name**  **Initial values** **Explanation**
    ---------------- ------------------ ---------------------------------------------------------    
    g                  0                  Synapse conductance on the post-synaptic neuron.

    s                  0                  Gating variable.

    x                  0                  Gating variable.                              
    ================ ================== =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    References:
        .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw. 
                "The Synapse." Principles of Computational Modelling in Neuroscience. 
                Cambridge: Cambridge UP, 2011. 172-95. Print.
    '''

    target_backend = 'general'

    def __init__(self, pre, post, conn, delay=0., tau1=1.0, tau2=3.0, **kwargs):
        # parameters
        self.tau1 = tau1
        self.tau2 = tau2
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.backend.shape(self.conn_mat)

        # variables
        self.s = bp.backend.zeros(self.size)
        self.x = bp.backend.zeros(self.size)
        self.w = bp.backend.ones(self.size) * .2
        self.out = self.register_constant_delay('out', size=self.size, delay_time=delay)

        super(Two_exponentials, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def integral(s, x, t, tau1, tau2):
        dxdt = (-(tau1 + tau2) * x - s) / (tau1 * tau2)
        dsdt = x
        return dsdt, dxdt
    
    def update(self, _t):
        self.s, self.x = self.integral(self.s, self.x, _t, self.tau1, self.tau2)
        self.x += bp.backend.reshape(self.pre.spike, (-1, 1)) * self.conn_mat
        self.out.push(self.w * self.s)
        self.post.input += bp.backend.sum(self.out.pull(), axis=0)

