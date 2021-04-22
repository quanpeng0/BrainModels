## 3.2 Firing rate networks

### 3.2.1 Decision model

In addition to spiking models, BrainPy can also implement Firing rate models. Let's first look at the implementation of a simplified version of the decision model. The model was simplified by the researcher (Wong & Wang, 2006) through a series of means such as mean field approach. In the end, there are only two variables, $$S_1$$ and $$S_2$$, which respectively represent the state of two neuron groups and correspond to two options.

<img src="../../figs/decision.png">

The formula is as follows:

$$
\frac{dS_1} {dt} = -\frac {S_1} \tau + (1-S_1) \gamma r_1
$$

$$
\frac{dS_2} {dt} = -\frac {S_2} \tau + (1-S_2) \gamma r_2
$$

Where $$r_1 $$ and $$r_2$$ is the firing rate of two neuron groups, which is given by the input-output function:

$$
r_i = f(I_{syn, i})
$$

$$
f(I)= \frac {aI-b} {1- \exp [-d(aI-b)]}
$$

The formula for the $$I_{syn, i}$$ is given by the model structure, which subtracts the suppressed current from the recurrent connection, and plus the background current and external input, which is available:

$$
I_{syn, 1} = J_{11} S_1 - J_{12} S_2 + I_0 + I_1
$$

$$
I_{syn, 2} = J_{22} S_2 - J_{21} S_1 + I_0 + I_2
$$


And the external inputs $$I_1, I_2$$ are determined by the total input strength $$\mu_0$$ and a coherence $$c'$$. The higher the coherence, the more definite $$S_1$$ is the correct answer, while the lower the coherence, the more random it is. The formula is as follows:

$$
I_1 = J_{\text{A, ext}} \mu_0 (1+\frac {c'}{100\%})
$$

$$
I_2 = J_{\text{A, ext}} \mu_0 (1-\frac {c'}{100\%})
$$



The code implementation is as follows: we can create a neuron group class, and use $$S_1$$ and $$S_2$$ to store the two states of the neuron group. The dynamics of the model can be implemented by a ``derivative`` function for dynamics analysis.


```python
class Decision(bp.NeuGroup):
    target_backend = 'general'

    @staticmethod
    def derivative(s1, s2, t, I, coh, JAext, J_rec, J_inh, I_0, b, d, a, tau_s, gamma):
        I1 = JAext * I * (1. + coh)
        I2 = JAext * I * (1. - coh)

        I_syn1 = J_rec * s1 - J_inh * s2 + I_0 + I1
        r1 = (a * I_syn1 - b) / (1. - bp.ops.exp(-d * (a * I_syn1 - b)))
        ds1dt = - s1 / tau_s + (1. - s1) * gamma * r1

        I_syn2 = J_rec * s2 - J_inh * s1 + I_0 + I2
        r2 = (a * I_syn2 - b) / (1. - bp.ops.exp(-d * (a * I_syn2 - b)))
        ds2dt = - s2 / tau_s + (1. - s2) * gamma * r2

        return ds1dt, ds2dt

    def __init__(self, size, coh, tau_s=.06, gamma=0.641,
                 J_rec=.3725, J_inh=.1137,
                 I_0=.3297, JAext=.00117,
                 a=270., b=108., d=0.154,
                 **kwargs):
        # parameters
        self.coh = coh
        self.tau_s = tau_s
        self.gamma = gamma
        self.J_rec = J_rec
        self.J_inh = J_inh
        self.I0 = I_0
        self.JAext = JAext
        self.a = a
        self.b = b
        self.d = d

        # variables
        self.s1 = bp.ops.ones(size) * .06
        self.s2 = bp.ops.ones(size) * .06
        self.input = bp.ops.zeros(size)

        self.integral = bp.odeint(f=self.derivative, method='rk4', dt=0.01)

        super(Decision, self).__init__(size=size, **kwargs)

    def update(self, _t):
        for i in prange(self.size):
            self.s1[i], self.s2[i] = self.integral(self.s1[i], self.s2[i], _t,
                                                   self.input[i], self.coh, self.JAext, self.J_rec,
                                                   self.J_inh, self.I0, self.b, self.d,
                                                   self.a, self.tau_s, self.gamma)
            self.input[i] = 0.
```

The code of phase plane analysis is as follows. Let's first look at the case when there is no external input. At this time, $$\mu_0 = 0$$.


```python
from collections import OrderedDict

pars = dict(tau_s=.06, gamma=0.641,
            J_rec=.3725, J_inh=.1137,
            I_0=.3297, JAext=.00117,
            b=108., d=0.154, a=270.)

pars['I'] = 0.
pars['coh'] = 0.

decision = Decision(1, coh=pars['coh'])

phase = bp.analysis.PhasePlane(decision.integral,
                               target_vars=OrderedDict(s2=[0., 1.], s1=[0., 1.]),
                               fixed_vars=None,
                               pars_update=pars,
                               numerical_resolution=.001,
                               options={'escape_sympy_solver': True})

phase.plot_nullcline()
phase.plot_fixed_point()
phase.plot_vector_field(show=True)
```

    plot nullcline ...
    plot fixed point ...
    Fixed point #1 at s2=0.06176109215560733, s1=0.061761097890810475 is a stable node.
    Fixed point #2 at s2=0.029354239100062428, s1=0.18815448592736211 is a saddle node.
    Fixed point #3 at s2=0.0042468423702408655, s1=0.6303045696241589 is a stable node.
    Fixed point #4 at s2=0.6303045696241589, s1=0.004246842370235128 is a stable node.
    Fixed point #5 at s2=0.18815439944520335, s1=0.029354240536530615 is a saddle node.
    plot vector field ...



![png](../../figs/out/output_77_1.png)


It can be seen that it is very convenient to use BrainPy for dynamics analysis. The vector field and fixed point indicate which option will fall in the end under different initial values.

Here, the x-axis is $$S_2$$ which represents choice 2, and the y-axis is $$S_1$$, which represents choice 1. As you can see, the upper-left fixed point represents choice 1, the lower-right fixed point represents choice 2, and the lower-left fixed point represents no choice.

Now let's see which option will eventually fall under different initial values with different coherence, and we fix the external input strength to 30.

Now let's look at the phase plane under different coherences when we fix the external input strength to 30.


```python
# coherence = 0%
pars['I'] = 30.
pars['coh'] = 0.

decision = Decision(1, coh=pars['coh'])

phase = bp.analysis.PhasePlane(decision.integral,
                               target_vars=OrderedDict(s2=[0., 1.], s1=[0., 1.]),
                               fixed_vars=None,
                               pars_update=pars,
                               numerical_resolution=.001,
                               options={'escape_sympy_solver': True})

print("coherence = 0%")
phase.plot_nullcline()
phase.plot_fixed_point()
phase.plot_vector_field(show=True)

# coherence = 51.2%
pars['coh'] = 0.512

decision = Decision(1, coh=pars['coh'])

phase = bp.analysis.PhasePlane(decision.integral,
                               target_vars=OrderedDict(s2=[0., 1.], s1=[0., 1.]),
                               fixed_vars=None,
                               pars_update=pars,
                               numerical_resolution=.001,
                               options={'escape_sympy_solver': True})

print("coherence = 51.2%")
phase.plot_nullcline()
phase.plot_fixed_point()
phase.plot_vector_field(show=True)


# coherence = 100%
pars['coh'] = 1.

decision = Decision(1, coh=pars['coh'])

phase = bp.analysis.PhasePlane(decision.integral,
                               target_vars=OrderedDict(s2=[0., 1.], s1=[0., 1.]),
                               fixed_vars=None,
                               pars_update=pars,
                               numerical_resolution=.001,
                               options={'escape_sympy_solver': True})

print("coherence = 100%")
phase.plot_nullcline()
phase.plot_fixed_point()
phase.plot_vector_field(show=True)
```

    coherence = 0%
    plot nullcline ...
    plot fixed point ...
    Fixed point #1 at s2=0.6993504413889349, s1=0.011622049526766405 is a stable node.
    Fixed point #2 at s2=0.49867489858358865, s1=0.49867489858358865 is a saddle node.
    Fixed point #3 at s2=0.011622051540013889, s1=0.6993504355529329 is a stable node.
    plot vector field ...



![png](../../figs/out/output_79_1.png)


    coherence = 51.2%
    plot nullcline ...
    plot fixed point ...
    Fixed point #1 at s2=0.5673124813731691, s1=0.2864701069327971 is a saddle node.
    Fixed point #2 at s2=0.6655747347157656, s1=0.027835279565912054 is a stable node.
    Fixed point #3 at s2=0.005397687847426814, s1=0.7231453520305031 is a stable node.
    plot vector field ...



![png](../../figs/out/output_79_3.png)


    coherence = 100%
    plot nullcline ...
    plot fixed point ...
    Fixed point #1 at s2=0.0026865954387078755, s1=0.7410985604497689 is a stable node.
    plot vector field ...



![png](../../figs/out/output_79_5.png)


### 3.2.2 CANN

Let's see another example of firing rate model, a continuous attractor neural network (CANN). The structure of one-dimensional CANN is as follows:

<img src="../../figs/cann.png">

We denote (x) as the parameter space site of the neuron group, and the dynamics of the total synaptic input of neuron group (x) $$u(x)$$ is given by:

$$
\tau \frac{du(x,t)}{dt} = -u(x,t) + \rho \int dx' J(x,x') r(x',t)+I_{ext}
$$

Where $$r(x', t)$$ is the firing rate of the neuron group (x'), which is given by:

$$
r(x,t) = \frac{u(x,t)^2}{1 + k \rho \int dx' u(x',t)^2}
$$

The intensity of excitatory connection between (x) and (x') $$J(x, x')$$ is given by a Gaussian function:

$$
J(x,x') = \frac{1}{\sqrt{2\pi}a}\exp(-\frac{|x-x'|^2}{2a^2})
$$

The external input $$I_{ext}$$ is related to position $$z(t)$$:

$$
I_{ext} = A\exp\left[-\frac{|x-z(t)|^2}{4a^2}\right]
$$


While implementing with BrainPy, we create a class of ``CANN1D`` by inheriting ``bp.NeuGroup`` as follow:


```python
class CANN1D(bp.NeuGroup):
    target_backend = ['numpy', 'numba']

    @staticmethod
    def derivative(u, t, conn, k, tau, Iext):
        r1 = np.square(u)
        r2 = 1.0 + k * np.sum(r1)
        r = r1 / r2
        Irec = np.dot(conn, r)
        du = (-u + Irec + Iext) / tau
        return du
    
    def __init__(self, num, tau=1., k=8.1, a=0.5, A=10., J0=4.,
                 z_min=-np.pi, z_max=np.pi, **kwargs):
        # parameters
        self.tau = tau  # The synaptic time constant
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # maximum connection value

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = np.linspace(z_min, z_max, num)  # The encoded feature values

        # variables
        self.u = np.zeros(num)
        self.input = np.zeros(num)

        # The connection matrix
        self.conn_mat = self.make_conn(self.x)

        self.int_u = bp.odeint(f=self.derivative, method='rk4', dt=0.05)
        
        super(CANN1D, self).__init__(size=num, **kwargs)

        self.rho = num / self.z_range  # The neural density
        self.dx = self.z_range / num  # The stimulus density

    def dist(self, d):
        d = np.remainder(d, self.z_range)
        d = np.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self, x):
        assert np.ndim(x) == 1
        x_left = np.reshape(x, (-1, 1))
        x_right = np.repeat(x.reshape((1, -1)), len(x), axis=0)
        d = self.dist(x_left - x_right)
        Jxx = self.J0 * np.exp(-0.5 * np.square(d / self.a)) / (np.sqrt(2 * np.pi) * self.a)
        return Jxx

    def get_stimulus_by_pos(self, pos):
        return self.A * np.exp(-0.25 * np.square(self.dist(self.x - pos) / self.a))

    def update(self, _t):
        self.u = self.int_u(self.u, _t, self.conn_mat, self.k, self.tau, self.input)
        self.input[:] = 0.

```

Where the functions ``dist`` and ``make_conn`` are designed to get the connection strength $$J$$ between each of the two neuron groups. In the ``make_conn`` function, we first calculate the distance matrix between each of the two $$x$$. Because neurons are arranged in rings, the value of $$x$$ is between $$-\pi$$ and $$\pi$$, so the range of $$x-x'$$ is $$2\pi$$, and -$$\pi$$ and $$\pi$$ are the same points (the actual furthest is $$\pi$$, that is, half of ``z_range``, the distance exceeded needs to be subtracted from a ``z_range``). We use the ``dist`` function to handle the distance on the ring.

The ``get_stimulus_by_pos`` function processes external inputs based on position ``pos``, which allows users to get input current by setting target positions. For example, in a simple population coding, we give an external input of ``pos=0``, and we run in this way:


```python
cann = CANN1D(num=512, k=0.1, monitors=['u'])

I1 = cann.get_stimulus_by_pos(0.)
Iext, duration = bp.inputs.constant_current([(0., 1.), (I1, 8.), (0., 8.)])
cann.run(duration=duration, inputs=('input', Iext))

bp.visualize.animate_1D(
    dynamical_vars=[{'ys': cann.mon.u, 'xs': cann.x, 'legend': 'u'},
                    {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
    frame_step=1,
    frame_delay=100,
    show=True,
    save_path='figs/CANN-encoding.gif'
)
```

<img src="../../figs/CANN-encoding.gif">

We can see that the shape of $$u$$ encodes the shape of external input.

Now we add random noise to the external input to see how the shape of $$u$$ changes.


```python
cann = CANN1D(num=512, k=8.1, monitors=['u'])

dur1, dur2, dur3 = 10., 30., 0.
num1 = int(dur1 / bp.backend.get_dt())
num2 = int(dur2 / bp.backend.get_dt())
num3 = int(dur3 / bp.backend.get_dt())
Iext = np.zeros((num1 + num2 + num3,) + cann.size)
Iext[:num1] = cann.get_stimulus_by_pos(0.5)
Iext[num1:num1 + num2] = cann.get_stimulus_by_pos(0.)
Iext[num1:num1 + num2] += 0.1 * cann.A * np.random.randn(num2, *cann.size)
cann.run(duration=dur1 + dur2 + dur3, inputs=('input', Iext))

bp.visualize.animate_1D(
    dynamical_vars=[{'ys': cann.mon.u, 'xs': cann.x, 'legend': 'u'},
                    {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
    frame_step=5,
    frame_delay=50,
    show=True,
    save_path='figs/CANN-decoding.gif'
)
```

<img src="../../figs/CANN-decoding.gif">

We can see that the shape of $$u$$ remains like a bell shape, which indicates that it can perform template matching based on the input.

Now let's give a moving input, we vary the position of the input with ``np.linspace``, we will see that the $$u$$ will follow the input, i.e., smooth tracking.


```python
cann = CANN1D(num=512, k=8.1, monitors=['u'])

dur1, dur2, dur3 = 20., 20., 20.
num1 = int(dur1 / bp.backend.get_dt())
num2 = int(dur2 / bp.backend.get_dt())
num3 = int(dur3 / bp.backend.get_dt())
position = np.zeros(num1 + num2 + num3)
position[num1: num1 + num2] = np.linspace(0., 12., num2)
position[num1 + num2:] = 12.
position = position.reshape((-1, 1))
Iext = cann.get_stimulus_by_pos(position)
cann.run(duration=dur1 + dur2 + dur3, inputs=('input', Iext))

bp.visualize.animate_1D(
    dynamical_vars=[{'ys': cann.mon.u, 'xs': cann.x, 'legend': 'u'},
                    {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
    frame_step=5,
    frame_delay=50,
    show=True,
    save_path='figs/CANN-tracking.gif'
)
```

<img src="../../figs/CANN-tracking.gif">