
## 1.3 Firing Rate models

Firing Rate models are simpler than reduced models. In these models, each compute unit represents a neuron group, the membrane potential variable $V$ in single neuron models is replaced by firing rate variable $a$ (or $r$ or $\nu$). Here we introduce a canonical firing rate unit.

### 1.3.1 Firing Rate Units

Wilson and Cowan (1972) proposed this unit to represent the activites in excitatory and inhibitory neuron columns. Each element of variables $a_e$ and $a_i$ refers to the average activity of a neuron column, which contains multiple neurons in each unit.

$$\tau_e \frac{d a_e(t)}{d t} = - a_e(t) + (k_e - r_e * a_e(t)) * \mathcal{S}_e(c_1 a_e(t) - c_2 a_i(t) + I_{ext_e}(t))$$

$$\tau_i \frac{d a_i(t)}{d t} = - a_i(t) + (k_i - r_i * a_i(t)) * \mathcal{S}_i(c_3 a_e(t) - c_4 a_i(t) + I_{ext_j}(t))$$

$$\mathcal{S}(x) = \frac{1}{1 + exp(- a(x - \theta))} - \frac{1}{1 + exp(a\theta)} $$


```python
class FiringRateUnit(bp.NeuGroup):
    target_backend = 'general'

    @staticmethod
    def derivative(a_e, a_i, t,
                   k_e, r_e, c1, c2, I_ext_e,
                   slope_e, theta_e, tau_e,
                   k_i, r_i, c3, c4, I_ext_i,
                   slope_i, theta_i, tau_i):
        daedt = (- a_e + (k_e - r_e * a_e) \
                 * mysigmoid(c1 * a_e - c2 * a_i + I_ext_e, slope_e, theta_e)) \
                / tau_e
        daidt = (- a_i + (k_i - r_i * a_i) \
                 * mysigmoid(c3 * a_e - c4 * a_i + I_ext_i, slope_i, theta_i)) \
                / tau_i
        return daedt, daidt

    def __init__(self, size, c1=12., c2=4., c3=13., c4=11.,
                 k_e=1., k_i=1., tau_e=1., tau_i=1., r_e=1., r_i=1.,
                 slope_e=1.2, slope_i=1., theta_e=2.8, theta_i=4.,
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
        self.input_e = bp.backend.zeros(size)
        self.input_i = bp.backend.zeros(size)
        self.a_e = bp.backend.ones(size) * 0.1
        self.a_i = bp.backend.ones(size) * 0.05

        self.integral = bp.odeint(self.derivative)
        super(FiringRateUnit, self).__init__(size=size, **kwargs)

    def mysigmoid(x, a, theta):
        return 1 / (1 + np.exp(- a * (x - theta))) \
               - 1 / (1 + np.exp(a * theta))

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
```
