## 1.1 Biophysical models

### 1.1.1 Hodgkin-Huxley model

Hodgkin and Huxley (1952) recorded the generation of action potential on squid giant axons with voltage clamp technique, and proposed the canonical neuron model called Hudgin-Huxley model (HH model). 

In last section we have introduced a general template for neuron membrane. Computational neuroscientists always model neuron membrane as equivalent circuit like the following figure.

<center><img src="../../figs/neurons/1-2.png">	</center>

<center><b>Fig. 1-4 Equivalent circuit diagram | NeuroDynamics </b></center>

The equivalent circuit diagram of Fig.1-1 is shown in Fig. 1-2, in which the battery $E_L$ refers to the potential difference across membrane, electric current $$I$$ refers to the external stimulus, capacitance $$C$$ refers to the hydrophobic phospholipid bilayer with low conductance, resistance $$R$$ refers to the resistance correspond to leaky current, i.e. the resistance of all non-specific ion channels. 

As Na+ ion channel and K+ ion channel are important in the generation of action potentials, these two ion channels are modeled as the two resistances $$R\_{Na}$$ and $$R_K$$ in parallel on the right side of the circuit diagram, and the two batteries $$E_{Na}$$ and $$E_K$$ refer to the ion potential differences caused by ion concentration differences of Na+ and K+, respectively.

Consider the Kirchhoff’s first law, that is,  for any node in an electrical circuit, the sum of currents flowing into that node is equal to the sum of currents flowing out of that node, Fig. 1-2 can be modeled as differential equations:
$$
C \frac{dV}{dt} = -(\bar{g}_{Na} m^3 h (V - E_{Na}) + \bar{g}_K n^4(V - E_K) + g_{leak}(V - E_{leak})) + I(t)
$$

$$
\frac{dx}{dt} = \alpha_x(1-x) - \beta_x , x \in \{ Na, K, leak \}
$$

```python
	@staticmethod
	def derivative(V, m, h, n, t, C, gNa, ENa, gK, EK, gL, EL, Iext):
    	dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    	dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    	dndt = alpha_n(V) * (1 - n) - beta_n(V) * n

    	I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
    	I_K = (gK * n ** 4.0) * (V - EK)
    	I_leak = gL * (V - EL)
    	dVdt = (- I_Na - I_K - I_leak + Iext) / C

    	return dVdt, dmdt, dhdt, dndt
```
That is the HH model. Note that in the first equation above, the first three terms on the right hand are the current go through Na+ ion channels, K+ ion channels and other non-specific ion channels, respectively, while $$I(t)$$ is an external input. On the left hand, $$C\frac{dV}{dt}$$ is the current go through the capacitance. 

In the computing of ion channel currents, other than the Ohm's law $$I = U/R = gR$$, HH model introduces three gating variables m, n and h to control the open/close state of ion channels. To be precise, variables m and h control the state of Na+ ion channel, variable n controls the state of K+ ion channel, and the real conductance of an ion channel is the product of maximal conductance $$\bar{g}$$ and the state of gating variables. Gating variables' dynamics can be expressed in Markov-like form, in which $$\alpha_x$$ refers to the activation rate of gating variable x, and $$\beta_x$$ refers to the de-activation rate of x. The expressions of $$\alpha_x$$ and $$\beta_x$$ (as shown in equations below) are fitted by experimental data.
$$
\alpha_m(V) = \frac{0.1(V+40)}{1 - exp(\frac{-(V+40)}{10})}
$$

$$
\beta_m(V) = 4.0 exp(\frac{-(V+65)}{18})
$$

$$
\alpha_h(V) = 0.07 exp(\frac{-(V+65)}{20})
$$

$$
\beta_h(V) = \frac{1}{1 + exp(\frac{-(V + 35)}{10})}
$$

$$
\alpha_n(V) = \frac{0.01(V+55)}{1 - exp(\frac{-(V+55)}{10})}
$$

$$
\beta_n(V) = 0.125 exp(\frac{-(V+65)}{80})
$$

```python
	def alpha_m(V): 
        return 0.1 * (V + 40) / (1 - bp.ops.exp(-(V + 40) / 10))
    
	def beta_m(V): 
        return 4.0 * bp.ops.exp(-(V + 65) / 18)
    
	def alpha_h(V): 
        return 0.07 * bp.ops.exp(-(V + 65) / 20.)
    
	def beta_h(V): 
        return 1 / (1 + bp.ops.exp(-(V + 35) / 10))
    
	def alpha_n(V): 
        return 0.01 * (V + 55) / (1 - bp.ops.exp(-(V + 55) / 10))
    
	def beta_n(V): 
        return 0.125 * bp.ops.exp(-(V + 65) / 80)
```
As HH model is computationally intensive and has few conditional judgment, exponential euler numerical integrationmethod is a simple but efficient one for this model. In BrainPy we may call `odeint` function provided by BrainPy in the constructor of `HH` class to specify the numerical method are to be used, and `odeint` function will recompile `derivative` method, transform the derivatives to variable update processes. 

*BrainPy supports the automatically integration of ODEs and SDEs on different methods, and will support PDEs etc. in the future. Users may call `brainpy.odeint` for ODEs and `brainpy.sdeint` for SDEs.*

```python
def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36.,
             EL=-54.387, gL=0.03, V_th=20., C=1.0, **kwargs):
    # model parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.C = C
    self.V_th = V_th

    # model variables
    num = bp.size2len(size)
    self.V = -65. * bp.ops.ones(num)
    self.m = 0.5 * bp.ops.ones(num)
    self.h = 0.6 * bp.ops.ones(num)
    self.n = 0.32 * bp.ops.ones(num)
    self.spike = bp.ops.zeros(num, dtype=bool)
    self.input = bp.ops.zeros(num)

    # def numerical solver
    self.integral = bp.odeint(f=self.derivative, method='exponential_euler')
    
    # super class init
    super(HH, self).__init__(size=size, **kwargs)
```
*Model variables like `V`, `m`, `h` are saved in memory as a floating point vector with length of neuron group `size`.*

In each time step of simulation, we may update the variables once, replace the old values $$x(t)$$ with $$x(t+1)$$. In HH model, we call `integral` method to update variables `V`, `m`, `n`, `h`, judge if there are spikes on each neuron and reset the external input of current moment in `update` method of HH class.

```python
def update(self, _t):
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t,
                      self.C, self.gNa, self.ENa, self.gK,
                      self.EK, self.gL, self.EL, self.input)
    self.spike = (self.V < self.V_th) * (V >= self.V_th)
    self.V = V
    self.m = m
    self.h = h
    self.n = n
    self.input[:] = 0
```




```python
import brainpy as bp

class HH(bp.NeuGroup):
    target_backend = 'general'

    @staticmethod
    def derivative(V, m, h, n, t, C, gNa, ENa, gK, EK, gL, EL, Iext):
        alpha = 0.1 * (V + 40) / (1 - bp.ops.exp(-(V + 40) / 10))
        beta = 4.0 * bp.ops.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m

        alpha = 0.07 * bp.ops.exp(-(V + 65) / 20.)
        beta = 1 / (1 + bp.ops.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h

        alpha = 0.01 * (V + 55) / (1 - bp.ops.exp(-(V + 55) / 10))
        beta = 0.125 * bp.ops.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n

        I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
        I_K = (gK * n ** 4.0) * (V - EK)
        I_leak = gL * (V - EL)
        dVdt = (- I_Na - I_K - I_leak + Iext) / C

        return dVdt, dmdt, dhdt, dndt

    def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36.,
                 EL=-54.387, gL=0.03, V_th=20., C=1.0, **kwargs):
        # parameters
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.C = C
        self.V_th = V_th

        # variables
        num = bp.size2len(size)
        self.V = -65. * bp.ops.ones(num)
        self.m = 0.5 * bp.ops.ones(num)
        self.h = 0.6 * bp.ops.ones(num)
        self.n = 0.32 * bp.ops.ones(num)
        self.spike = bp.ops.zeros(num, dtype=bool)
        self.input = bp.ops.zeros(num)

        # numerical solver
        self.integral = bp.odeint(f=self.derivative, method='exponential_euler')
        super(HH, self).__init__(size=size, **kwargs)

    def update(self, _t):
        V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t,
                                   self.C, self.gNa, self.ENa, self.gK,
                                   self.EK, self.gL, self.EL, self.input)
        self.spike = (self.V < self.V_th) * (V >= self.V_th)
        self.V = V
        self.m = m
        self.h = h
        self.n = n
        self.input[:] = 0
```


```python
import brainpy as bp

neu = HH(100, monitors=['V'])
net = bp.Network(neu)
net.run(200., inputs=(neu, 'input', 5.))

bp.visualize.line_plot(neu.mon.ts, neu.mon.V, show=True)
```


![png](../../figs/neurons/out/output_27_0.png)

The V-t plot of HH model simulated by BrainPy is shown above. The three periods, depolarization, repolarization and refractory period of a real action potential can be seen in the V-t plot. In addition, during the depolarization period, the membrane integrates external inputs slowly at first, and increases rapidly once it grows beyond some point, which also reproduces the "shape" of action potentials.

All these features can be mapped to the equations one by one. The slow variable h remains small and will not be activated in a time period after spike, which can be seen as refractory period.

加链接：See more details in [somewhere].