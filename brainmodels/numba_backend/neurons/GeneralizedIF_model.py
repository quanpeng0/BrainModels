# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np
from numba import prange


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
        for i in prange(self.size[0]):
            I1, I2, V_th, V = self.integral(
                self.I1[i], self.I2[i], self.V_th[i], self.V[i], _t, 
                self.k1, self.k2, self.a, self.V_rest,
                self.b, self.V_th_inf, 
                self.R, self.input[i], self.tau
            )
            self.spike[i] = self.V_th[i] < V
            if self.spike[i]:
                V = self.V_reset
                I1 = self.R1 * I1 + self.A1
                I2 = self.R2 * I2 + self.A2
                V_th = max(V_th, self.V_th_reset)
            self.I1[i] = I1
            self.I2[i] = I2
            self.V_th[i] = V_th
            self.V[i] = V
        self.input[:] = 0.
        
'''  
import matplotlib.pyplot as plt
import brainpy as bp

# set parameters
num2mode = ["tonic_spiking",       "class_1",                           "spike_frequency_adaptation",
            "phasic_spiking",      "accomodation",                      "threshold_variability",
            "rebound_spike",       "class_2",                           "integrator",
            "input_bistability",   "hyperpolarization_induced_spiking", "hyperpolarization_induced_bursting",
            "tonic_bursting",      "phasic_bursting",                   "rebound_burst",
            "mixed_mode",          "afterpotentials",                   "basal_bistability",
            "preferred_frequency", "spike_latency"]

mode2param = {
    "tonic_spiking": {
        "input": [(1.5, 200.)]
    },
    "class_1": {
        "input": [(1. + 1e-6, 500.)]
    },
    "spike_frequency_adaptation": {
        "a": 0.005, "input": [(2., 200.)]
    },
    "phasic_spiking": {
        "a": 0.005, "input": [(1.5, 500.)]
    },
    "accomodation": {
        "a": 0.005,
        "input": [(1.5, 100.), (0, 500.), (0.5, 100.),
                  (1., 100.), (1.5, 100.), (0., 100.)]
    },
    "threshold_variability": {
        "a": 0.005,
        "input": [(1.5, 20.), (0., 180.), (-1.5, 20.),
                  (0., 20.), (1.5, 20.), (0., 140.)]
    },
    "rebound_spike": {
        "a": 0.005,
        "input": [(0, 50.), (-3.5, 750.), (0., 200.)]
    },
    "class_2": {
        "a": 0.005,
        "input": [(2 * (1. + 1e-6), 200.)],
        "V_th": -30.
    },
    "integrator": {
        "a": 0.005,
        "input": [(1.5, 20.), (0., 10.), (1.5, 20.), (0., 250.),
                  (1.5, 20.), (0., 30.), (1.5, 20.), (0., 30.)]
    },
    "input_bistability": {
        "a": 0.005,
        "input": [(1.5, 100.), (1.7, 400.),
                  (1.5, 100.), (1.7, 400.)]
    },
    "hyperpolarization_induced_spiking": {
        "V_th_reset": -60.,
        "V_th_inf": -120.,
        "input": [(-1., 400.)],
        "V_th": -50.
    },
    "hyperpolarization_induced_bursting": {
        "V_th_reset": -60.,
        "V_th_inf": -120.,
        "A1": 10.,
        "A2": -0.6,
        "input": [(-1., 400.)],
        "V_th": -50.
    },
    "tonic_bursting": {
        "a": 0.005,
        "A1": 10.,
        "A2": -0.6,
        "input": [(2., 500.)]
    },
    "phasic_bursting": {
        "a": 0.005,
        "A1": 10.,
        "A2": -0.6,
        "input": [(1.5, 500.)]
    },
    "rebound_burst": {
        "a": 0.005,
        "A1": 10.,
        "A2": -0.6,
        "input": [(0, 100.), (-3.5, 500.), (0., 400.)]
    },
    "mixed_mode": {
        "a": 0.005,
        "A1": 5.,
        "A2": -0.3,
        "input": [(2., 500.)]
    },
    "afterpotentials": {
        "a": 0.005,
        "A1": 5.,
        "A2": -0.3,
        "input": [(2., 15.), (0, 185.)]
    },
    "basal_bistability": {
        "A1": 8.,
        "A2": -0.1,
        "input": [(5., 10.), (0., 90.), (5., 10.), (0., 90.)]
    },
    "preferred_frequency": {
        "a": 0.005,
        "A1": -3.,
        "A2": 0.5,
        "input": [(5., 10.), (0., 10.), (4., 10.), (0., 370.),
                  (5., 10.), (0., 90.), (4., 10.), (0., 290.)]
    },
    "spike_latency": {
        "a": -0.08,
        "input": [(8., 2.), (0, 48.)]
    }
}
    
def run_GIF_with_mode(mode = 'tonic_spiking', size = 10.,
                      row_p = 0, col_p = 0, fig = None, gs = None):
    
    print(f"Running GIF neuron neu with mode '{mode}'")
    #neu = brainmodels.neurons.GeneralizedIF(size, monitors = ['V', 'V_th', 'I1', 'I2', 'input'])
    neu = GeneralizedIF(size, monitors = ['V', 'V_th', 'I1', 'I2', 'input'])
    param = mode2param[mode].items()
    member_type = 0
    for (k, v) in param:
        if k == 'input':
            I_ext, dur = bp.inputs.constant_current(v)
            member_type += 1
        else:
            if member_type==0:
                exec("neu.%s = %f"%(k, v))
            else:
                exec("neu.%s = bp.backend.ones(size) * %f"%(k, v))
    neu.run(dur, inputs = ('input', I_ext), report = False)

    ts = neu.mon.ts
    ax1 = fig.add_subplot(gs[row_p, col_p])
    ax1.title.set_text(f'{mode}')

    ax1.plot(ts, neu.mon.V[:, 0], label='V')
    ax1.plot(ts, neu.mon.V_th[:, 0], label='V_th')
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

size = 10
pattern_num = 20
row_b = 2
col_b = 2
size_b = row_b * col_b
for i in range(pattern_num):
    if i % size_b == 0:
        fig, gs = bp.visualize.get_figure(row_b, col_b, 4, 8)
    mode = num2mode[i]
    run_GIF_with_mode(mode = mode, size = size,
                      row_p = i % size_b // col_b,
                      col_p = i % size_b % col_b,
                      fig = fig, gs = gs)
    if (i+1) % size_b == 0:
        plt.show()'''
