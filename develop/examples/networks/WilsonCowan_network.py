import brainpy as bp

class FiringRateUnit(bp.NeuGroup):
    target_backend = 'general'

    @staticmethod
    @bp.odeint
    def integral(a_e, a_i, t,
                 k_e, r_e, c1, c2,
                 I_ext_e, slope_e, theta_e, tau_e,
                 k_i, r_i, c3, c4,
                 I_ext_i, slope_i, theta_i, tau_i
                 ):
        x_ae = c1 * a_e - c2 * a_i + I_ext_e
        sigmoid_ae_l = 1 + bp.ops.exp(- slope_e *
                                      (x_ae - theta_e))
        sigmoid_ae_r = 1 + bp.ops.exp(slope_e * theta_e)
        sigmoid_ae = 1 / sigmoid_ae_l - 1 / sigmoid_ae_r
        daedt = (- a_e + (k_e - r_e * a_e) * sigmoid_ae) \
                / tau_e

        x_ai = c3 * a_e - c4 * a_i + I_ext_i
        sigmoid_ai_l = 1 + bp.ops.exp(- slope_i *
                                      (x_ai - theta_i))
        sigmoid_ai_r = 1 + bp.ops.exp(slope_i * theta_i)
        sigmoid_ai = 1 / sigmoid_ai_l - 1 / sigmoid_ai_r
        daidt = (- a_i + (k_i - r_i * a_i) * sigmoid_ai) \
                / tau_i
        return daedt, daidt

    def __init__(self, size, c1=12., c2=4., c3=13., c4=11.,
                 k_e=1., k_i=1., tau_e=1., tau_i=1.,
                 r_e=1., r_i=1., slope_e=1.2, slope_i=1.,
                 theta_e=2.8, theta_i=4.,
                 **kwargs):
        # params
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.k_e = k_e
        self.k_i = k_i
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.r_e = r_e
        self.r_i = r_i
        self.slope_e = slope_e
        self.slope_i = slope_i
        self.theta_e = theta_e
        self.theta_i = theta_i

        # vars
        self.input_e = bp.ops.zeros(size)
        self.input_i = bp.ops.zeros(size)
        self.a_e = bp.ops.ones(size) * 0.1
        self.a_i = bp.ops.ones(size) * 0.05

        super(FiringRateUnit, self).__init__(size=size, **kwargs)

    def update(self, _t):
        self.a_e, self.a_i = self.integral(
            self.a_e, self.a_i, _t,
            self.k_e, self.r_e, self.c1, self.c2,
            self.input_e, self.slope_e,
            self.theta_e, self.tau_e,
            self.k_i, self.r_i, self.c3, self.c4,
            self.input_i, self.slope_i,
            self.theta_i, self.tau_i)
        self.input_e[:] = 0.
        self.input_i[:] = 0.


import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt

# define params
bp.backend.set(dt=0.1)
step_dur = 10.  # duration for each P value, 10ms is basically enough to reach steady state
step_num = 100

#build network
neu = FiringRateUnit(1, monitors=['a_e', 'a_i'])
net = bp.Network(neu)

def plot_EP(net, I_sval, I_eval, step_num, fig, label, show):

    I_list = np.linspace(I_sval, I_eval, step_num, endpoint = False)
    input_list = []
    for I in I_list:
        input_list.append((I, step_dur))
    inputs, dur = bp.inputs.constant_input(input_list)
    ## simulate
    net.run(duration = dur, inputs = (neu, 'input_e', inputs.T))
    ## record a_e at each step
    a_e_list = []
    for i in range(step_num):
        a_e_list.append(neu.mon.a_e[i * 100, :])
    a_e_list = np.array(a_e_list)
    ## plot fig
    fig.add_subplot(gs[0, 0])
    print(neu.mon.a_e.shape)
    plt.plot(I_list, a_e_list, label = label)
    plt.xlabel("input_e")
    plt.ylabel("a_e")
    plt.legend()
    if show==True:
        plt.show()

# input_e increase
neu.a_e = -0.02
I_sval = -0.6
I_eval = 0.6
fig, gs = bp.visualize.get_figure(1, 1, 4, 4)
plot_EP(net, I_sval, I_eval, step_num, fig, label="raising", show=False)
# input_e decrease
neu.a_e = 0.48
I_sval = 0.6
I_eval = -0.6
plot_EP(net, I_sval, I_eval, step_num, fig, label="falling", show=True)