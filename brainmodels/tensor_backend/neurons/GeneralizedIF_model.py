# -*- coding: utf-8 -*-

import brainpy as bp

bp.backend.set('numpy', dt=0.02)

class GeneralizedIF(bp.NeuGroup):
    """
    Generalized Integrate-and-Fire model (GeneralizedIF model).

    .. math::

        &\\frac{d I_j}{d t} = - k_j I_j

        &\\frac{d V}{d t} = ( - (V - V_{rest}) + R\\sum_{j}I_j + RI) / \\tau

        &\\frac{d V_{th}}{d t} = a(V - V_{rest}) - b(V_{th} - V_{th\\infty})

    When V meet Vth, Generalized IF neuron fire:

    .. math::

        &I_j \\leftarrow R_j I_j + A_j

        &V \\leftarrow V_{reset}

        &V_{th} \\leftarrow max(V_{th_{reset}}, V_{th})

    Note that I_j refers to arbitrary number of internal currents.

    **Neuron Parameters**

    ============= ============== ======== ====================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------------------------------------------
    V_rest        -70.           mV       Resting potential.

    V_reset       -70.           mV       Reset potential after spike.

    V_th_inf      -50.           mV       Target value of threshold potential V_th updating.

    V_th_reset    -60.           mV       Free parameter, should be larger than V_reset.

    R             20.            \        Membrane resistance.

    tau           20.            \        Membrane time constant. Compute by R * C.

    a             0.             \        Coefficient describes the dependence of 

                                          V_th on membrane potential.

    b             0.01           \        Coefficient describes V_th update.

    k1            0.2            \        Constant pf I1.

    k2            0.02           \        Constant of I2.

    R1            0.             \        Free parameter. 

                                          Describes dependence of I_1 reset value on 

                                          I_1 value before spiking.

    R2            1.             \        Free parameter. 

                                          Describes dependence of I_2 reset value on 

                                          I_2 value before spiking.

    A1            0.             \        Free parameter.

    A2            0.             \        Free parameter.

    noise         0.             \        noise.

    mode          'scalar'       \        Data structure of ST members.
    ============= ============== ======== ====================================================================

    **Neuron Variables**    

    An object of neuron class record those variables for each neuron:

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                  -70.              Membrane potential.

    input              0.                External and synaptic input current.

    spike              0.                Flag to mark whether the neuron is spiking. 

                                         Can be seen as bool.

    V_th               -50.              Spiking threshold potential.

    I1                 0.                Internal current 1.

    I2                 0.                Internal current 2.

    t_last_spike       -1e7              Last spike time stamp.
    ================== ================= ==============================================

    References:
        .. [1] Mihalaş, Ştefan, and Ernst Niebur. "A generalized linear 
               integrate-and-fire neural model produces diverse spiking 
               behaviors." Neural computation 21.3 (2009): 704-718.
    """

    target_backend = 'general'

    def __init__(self, size, V_rest=-70., V_reset=-70., 
                 V_th_inf=-50., V_th_reset=-60., R=20., tau=20., 
                 a=0., b=0.01, k1=0.2, k2=0.02, 
                 R1=0., R2=1., A1=0., A2=0.,
                 **kwargs):
        #params
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th_inf = V_th_inf
        self.V_th_reset = V_th_reset
        self.R = R
        self.tau = tau
        self.a = a
        self.b = b
        self.k1 = k1
        self.k2 = k2
        self.R1 = R1
        self.R2 = R2
        self.A1 = A1
        self.A2 = A2

        #vars
        self.input = bp.backend.zeros(size)
        self.spike = bp.backend.zeros(size, dtype = bool)
        self.I1 = bp.backend.zeros(size)
        self.I2 = bp.backend.zeros(size)
        self.V = bp.backend.ones(size) * -70.
        self.V_th = bp.backend.ones(size) * -50.

        super(GeneralizedIF, self).__init__(size = size, **kwargs)
    
    @staticmethod
    @bp.odeint()
    def integral(I1, I2, V_th, V, t, 
                 k1, k2, a, V_rest, b, V_th_inf,
                 R, I_ext, tau):
        dI1dt = - k1 * I1
        dI2dt = - k2 * I2
        dVthdt = a * (V - V_rest) - b * (V_th - V_th_inf)
        dVdt = (- (V - V_rest) + R * I_ext + R * I1 + R * I2) / tau
        return dI1dt, dI2dt, dVthdt, dVdt
    
    def update(self, _t):
        I1, I2, V_th, V = self.integral(
            self.I1, self.I2, self.V_th, self.V, _t, 
            self.k2, self.k2, self.a, self.V_rest,
            self.b, self.V_th_inf, 
            self.R, self.input, self.tau)
        sp = (self.V_th < V)
        V[sp] = self.V_reset
        I1[sp] = self.R1 * I1[sp] + self.A1
        I2[sp] = self.R2 * I2[sp] + self.A2
        reset_th = np.logical_and(V_th < self.V_th_reset, sp)
        V_th[reset_th] = self.V_th_reset
        self.spike = sp
        self.I1 = I1
        self.I2 = I2
        self.V_th = V_th
        self.V = V
        self.input[:] = 0.

if __name__ == "__main__":

    size = 10
    group = GeneralizedIF(size, monitors = ['input', 'V', 'V_th'])
    duration = 200.
    # simulate
    mode = "basal_bistability"
    print(f"Choose parameters fit for <{mode}> mode")
    if mode == 'tonic_spiking':
        I_ext, dur = bp.inputs.constant_current([(1.5, duration)])
        group.run(duration=dur, inputs=["input", I_ext], report=True)
    elif mode == "class_1":
        I_ext, dur = bp.inputs.constant_current([(1.+1e-6, 500.)])
        group.run(duration=dur, inputs=["input", I_ext], report=True)
    elif mode == "spike_frequency_adaptation":
        group.a = 0.005
        I_ext, dur = bp.inputs.constant_current([(2., duration)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "phasic_spiking":
        group.a = 0.005
        I_ext, dur = bp.inputs.constant_current([(1.5, 500.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "accomodation":
        group.a = 0.005
        I_ext, dur = bp.inputs.constant_current([(1.5, 100.), (0, 500.), (0.5, 100.), 
                                                (1., 100.), (1.5, 100.), (0., 100.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "threshold_variability":
        group.a = 0.005
        I_ext, dur = bp.inputs.constant_current(
                        [(1.5, 20.), (0., 180.), (-1.5, 20.), 
                        (0., 20.), (1.5, 20.), (0., 140.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "rebound_spike":
        group.a = 0.005
        I_ext, dur = bp.inputs.constant_current(
                        [(0, 50.), (-3.5, 750.), (0., 200.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "class_2":
        group.a = 0.005
        group.V_th = bp.backend.ones(size) * -30.
        I_ext, dur = bp.inputs.constant_current([(2 * (1. + 1e-6), duration)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "integrator":
        group.a = 0.005
        I_ext, dur = bp.inputs.constant_current(
                        [(1.5, 20.), (0., 10.), (1.5, 20.), (0., 250.),
                        (1.5, 20.), (0., 30.), (1.5, 20.), (0., 30.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "input_bistability":
        group.a = 0.005
        I_ext, dur = bp.inputs.constant_current(
                        [(1.5, 100.), (1.7, 400.), 
                        (1.5, 100.), (1.7, 400.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "hyperpolarization_induced_spiking":
        group.V_th = bp.backend.ones(size) * -50.
        group.V_th_reset = -60.
        group.V_th_inf = -120.
        I_ext, dur = bp.inputs.constant_current([(-1., 400.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "hyperpolarization_induced_bursting":
        group.V_th = bp.backend.ones(size) * -50.
        group.V_th_reset = -60.
        group.V_th_inf = -120.
        group.A1 = 10 
        group.A2 = -0.6
        I_ext, dur = bp.inputs.constant_current([(-1., 400.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "tonic_bursting":
        group.a = 0.005
        group.A1 = 10
        group.A2 = -0.6
        I_ext, dur = bp.inputs.constant_current([(2., 500.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "phasic_bursting":
        group.a = 0.005
        group.A1 = 10
        group.A2 = -0.6
        I_ext, dur = bp.inputs.constant_current([(1.5, 500.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "rebound_burst":
        group.a = 0.005
        group.A1 = 10
        group.A2 = -0.6
        I_ext, dur = bp.inputs.constant_current(
                        [(0, 100.), (-3.5, 500.), (0., 400.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "mixed_mode":
        group.a = 0.005
        group.A1 = 5
        group.A2 = -0.3
        I_ext, dur = bp.inputs.constant_current([(2., 500.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "afterpotentials":
        group.a = 0.005
        group.A1 = 5
        group.A2 = -0.3
        I_ext, dur = bp.inputs.constant_current(
                        [(2., 15.), (0, 185.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "basal_bistability":
        group.A1 = 8
        group.A2 = -0.1
        I_ext, dur = bp.inputs.constant_current(
                        [(5., 10.), (0., 90.), (5., 10.), (0., 90.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "preferred_frequency":
        group.a = 0.005
        group.A1 = -3
        group.A2 = 0.5
        I_ext, dur = bp.inputs.constant_current(
                        [(5., 10.), (0., 10.), (4., 10.), (0., 370.), 
                        (5., 10.), (0., 90.), (4., 10.), (0., 290.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    elif mode == "spike_latency":
        group.a = -0.08
        I_ext, dur = bp.inputs.constant_current(
                        [(8., 2.), (0, 48.)])
        group.run(duration = dur, inputs = ["input", I_ext], report=True)
    else:
        raise ValueError(f"Error: Mode {mode} is not supported!")
    
    
    fig, gs = bp.visualize.get_figure(1, 1, 4, 8)
    ts = group.mon.ts
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.title.set_text(f'{mode}')

    ax1.plot(ts, group.mon.V[:, 0], label='V')
    ax1.plot(ts, group.mon.V_th[:, 0], label='V_th')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane potential')
    ax1.set_xlim(-0.1, ts[-1] + 0.1)
    plt.legend()

    ax2 = ax1.twinx()
    ax2.plot(ts, I_ext, color = 'turquoise', label='input')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('External input')
    ax2.set_xlim(-0.1, ts[-1] + 0.1)
    ax2.set_ylim(-5., 20.)
    plt.legend(loc = 'lower left')
    plt.show()
    '''
    bp.visualize.line_plot(group.mon.ts, group.mon.input)
    bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)
    '''