# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: brainmodels
#     language: python
#     name: brainmodels
# ---

# %% [markdown]
# # Hodgkin-Huxley Model

# %% [markdown]
# ## Model Overview

# %% [markdown]
# Hodgkin-Huxley model was proposed by Hodgkin and Huxley in 1952. This model provides a mathematical description of the initiation and propagation of the action potentials in neurons.
#
# The ionic mechanisms are described by an electrical circuit:
#
# <img src="../../images/HH-circuit.png">
#
# The lipid bilayer of the membrane is represented as a capacitance ($C$). Resistors, $R_{Na}$, $R_K$ and $R$ represent sodium, potassium and unspecific ion channels, respectively. The batteries ($E_{Na}, E_K, E_L$) represent ion reversal potentials based on the ion concentration difference. The reversal potentials can be calculated from the Nernst equations. $E_{Na}$, $E_{K}$, and $E_{L}$ under $37^\circ \rm C$ are around $+50$mV, $-77$mV, and $-54.387$mV, respectively.

# %% [markdown]
# ## BrainPy Implementation

# %% [markdown]
# ### Dynamics of membrane potential

# %% [markdown]
# Let's take a look at the mathematical expression of the model dynamics. We know from the electrical circuit:
#
# $$ C \frac {dV} {dt} = -(I_{Na} + I_{K} + I_{leak}) + I(t) $$
#
# that is: 
#
# $$ C \frac {dV} {dt} = - \sum_k I_{ion, k} + I(t) $$
#
#
# Using Ohm's law (I=gV), we get:
#
# $$ I_x = g_x (V - E_x) , \quad x\in \{Na, K, leak \} $$
#
# where $x$ is a specific ion channel.
#
# Experiments suggest that $g_{Na}$ and $g_K$ are functions of time and membrane potential, reflecting the change of permeability.
#
# The formal assumptions used to describe the sodium and potassium conductances are:
#
# $$ g_{Na} = \bar{g}_{Na} m^3 h,   \qquad  m, h \in [ 0, 1 ]$$ 
#
# $$ g_K = \bar{g}_K n^4,   \qquad  n \in [ 0, 1 ] $$
#
# where $\bar{g}_{Na}$ and $\bar{g}_{K}$ are constants representing the maximum conductances. $m$ and $h$ are dimensionless variables indicating the state of sodium channels, and $n$ represents the state of potassium channels. In other words, $x=1$ means the channel is open and $x=0$ corresponds to the channel being closed.

# %% [markdown]
# ### Dynamics of ionic channels

# %% [markdown]
# The dynamics of gating variables are given by:
#
# $$
# \begin{aligned}
# \frac {dx} {dt} & =  - \frac {x - x_0(V)} {\tau_x (V)} \\
# & =  \alpha_x (1-x)  - \beta_x x
# \end{aligned}
# $$ 
#
# where $x$ can be $m, n, h$, and $\alpha$ and $\beta$ are rate constants.
#
# The functions $\alpha$ and $\beta$ are given by:
#
# $$
# \begin{aligned}
# \alpha_m(V) &= \frac {0.1(V+40)}{1-\exp(\frac{-(V + 40)} {10})} \\
# \beta_m(V) &= 4.0 \exp(\frac{-(V + 65)} {18}) \\
# \alpha_h(V) &= 0.07 \exp(\frac{-(V+65)}{20}) \\
# \beta_h(V) &= \frac 1 {1 + \exp(\frac{-(V + 35)} {10})} \\
# \alpha_n(V) &= \frac {0.01(V+55)}{1-\exp(-(V+55)/10)} \\
# \beta_n(V) &= 0.125 \exp(\frac{-(V + 65)} {80}) 
# \end{aligned}
# $$

# %%
import brainmodels
import brainpy as bp


# %% [markdown]
# ### Simulation

# %% [markdown]
# Let's start simulation.

# %%
N = (100,)

neuron = brainmodels.neurons.HH(N, monitors=['spike', 'V', 'm', 'h', 'n'])

# set initial variable state
neuron.V[:] = bp.math.random.random(N) * 20 + -75

# %% [markdown]
# We can generate a step current by using `bp.inputs.constant_current`.

# %%
# step current parameters
amplitude = 6.9  # step current amplitude
stim_start = 10. # stimulation start point
stim_t = 50.   # stimulation duration
post_stim_t = 20. # after stimulation time

# generate step current
I = [(0, stim_start), (amplitude, stim_t), (0, post_stim_t)]
(step_I, duration) = bp.inputs.constant_current(I)

# apply input current to the neuron
neuron.run(duration=duration, inputs=('input', step_I, 'iter'))
bp.visualize.line_plot(neuron.mon.ts, neuron.mon.V, show=True)

# %% [markdown]
# ## Model Visualization

# %% [markdown]
# ### Dynamics of membrane potential

# %%
import matplotlib.pyplot as plt

# read time sequence
ts = neuron.mon.ts
fig, gs = bp.visualize.get_figure(3, 1, 3, 8)

# plot input current
fig.add_subplot(gs[0, 0])
plt.plot(ts, step_I, 'r')
plt.ylabel('Input Current ($\mu$A)')
plt.xlim(-0.1, duration + 0.1)
plt.xlabel('Time (ms)')
plt.title('Input Current')

# plot membrane potential
fig.add_subplot(gs[1, 0])
plt.plot(ts, neuron.mon.V[:, 0])
plt.ylabel('Membrane potential (mV)')
plt.xlim(-0.1, duration + 0.1)
plt.title('Membrane potential')

# plot gate variables
fig.add_subplot(gs[2, 0])
plt.plot(ts, neuron.mon.m[:, 0], label='Na active (m)')
plt.plot(ts, neuron.mon.h[:, 0], label='Na inactive (h)')
plt.plot(ts, neuron.mon.n[:, 0], label='K (n)')
plt.legend()
plt.xlim(-0.1, duration + 0.1)
plt.xlabel('Time (ms)')
plt.ylabel('gate variables')
plt.title('gate variables')

plt.show()

# %% [markdown]
# We can see from the graphs that the results corresponse to neuro-phisiological recordings. We can see depolarization, repolarization and hyperpolarization stages from the simulated action potential.
#
# The graph of gate variables demonstrate that $m$ increased during depolarization, which indicates the opening of $Na^+$ channels ($Na^+$ influx). While $n$ reached it's peak during repolarization stage, which reflects the opening of $K^+$ channels ($K^+$ efflux). The raise of $h$ and drop of $Na$ during repolarization result in the closing of $Na^+$ channels.

# %% [markdown]
# ### Channel currents and conductances

# %%
# conductances of ion channels
g_Na = neuron.gNa * bp.math.power(neuron.mon.m[:, 0], 3.0) * neuron.mon.h[:, 0]
g_K = neuron.gK * bp.math.power(neuron.mon.n[:, 0], 4.0)
V = neuron.mon.V[:, 0]
I_Na = g_Na * (V - neuron.ENa)
I_K = g_K * (V - neuron.EK)
I_leak = neuron.gL * (V - neuron.EL)

fig, gs = bp.visualize.get_figure(2, 1, 3, 8)

# plot input current
fig.add_subplot(gs[0, 0])
plt.plot(ts, I_Na, label = 'Na')
plt.plot(ts, I_K, label = 'K')
plt.plot(ts, I_leak, label = 'leak')
plt.legend()
plt.ylabel('Current ($\mu$A)')
plt.xlim(-0.1, duration + 0.1)
plt.xlabel('Time (ms)')
plt.title('Current')

# plot gate variables
fig.add_subplot(gs[1, 0])
#plt.plot(ts, V, label='V')
plt.plot(ts, g_Na, label='Na')
plt.plot(ts, g_K, label='K')
plt.legend()
plt.xlim(-0.1, duration + 0.1)
plt.xlabel('Time (ms)')
plt.ylabel('conductance(m.mho/cm$^2$)')
plt.title('channel conductance')

plt.show()

# %% [markdown]
# ### Gate variables

# %%
# plot n, m, h of v.
V = neuron.mon.V[:, 0]

plt.figure()
plt.plot(V, neuron.mon.n[:, 0], 'b', label='n')
plt.plot(V, neuron.mon.m[:, 0], 'r', label='m')
plt.plot(V, neuron.mon.h[:, 0], 'g', label='h')
plt.ylabel('Gate variables')
plt.xlabel("Membrane Voltage (mV)")
plt.title('Gate variables as functions of V')
plt.legend()

# %% [markdown]
# ### Rate constants

# %% [markdown]
# $\alpha_x$ and $\beta_x$ represent the opening and closing rate of channel $x$, respectively. Now let's take a look at the change of rate constants as the voltage increases.

# %%
# plot alpha and beta values of v.
alpha_m = 0.1 * (V + 40) / (1 - bp.math.exp(-(V + 40) / 10))
beta_m = 4.0 * bp.math.exp(-(V + 65) / 18)

alpha_h = 0.07 * bp.math.exp(-(V + 65) / 20.)
beta_h = 1 / (1 + bp.math.exp(-(V + 35) / 10))

alpha_n = 0.01 * (V + 55) / (1 - bp.math.exp(-(V + 55) / 10))
beta_n = 0.125 * bp.math.exp(-(V + 65) / 80)

fig, gs = bp.visualize.get_figure(3, 1, 3, 5)

# plot n
fig.add_subplot(gs[0, 0])
plt.plot(V, alpha_n, 'b', label=r'$\alpha_n$')
plt.plot(V, beta_n, 'r', label=r'$\beta_n$')
plt.ylabel('Rate constant')
plt.xlabel("Membrane Voltage (mV)")
plt.title('K Channel Rate constant of V')
plt.legend()

# plot m
fig.add_subplot(gs[1, 0])
plt.plot(V, alpha_m, 'b', label=r'$\alpha_m$')
plt.plot(V, beta_m, 'r', label=r'$\beta_m$')
plt.ylabel('Rate constant')
plt.xlabel("Membrane Voltage (mV)")
plt.title('Na channel (active) Rate constant of V')
plt.legend()

# plot h
fig.add_subplot(gs[2, 0])
plt.plot(V, alpha_h, 'b', label=r'$\alpha_h$')
plt.plot(V, beta_h, 'r', label=r'$\beta_h$')
plt.ylabel('Rate constant')
plt.xlabel("Membrane Voltage (mV)")
plt.title('Na channel (inactive) Rate constant of V')
plt.legend()

# %% [markdown]
# The graphs show that $\alpha$ increases and $\beta$ decreases as the increment of voltage for $m$ ($Na^+$ active) and $n$ ($K^+$), which modulates the opening of ion channels.
#
# While for $h$ ($Na^+$ inactive), the $\alpha$ decrease and $\beta$ increase as the increment of voltage, which modulates the closing of $Na^+$ channels.

# %% [markdown]
# ### Steady values

# %% [markdown]
# A fixed point of gate variable of channel $x$, $x_\infty$, where the ion channels reach their steady state (equilibrium), is given by
#
# $$ x_{\infty} = \frac {\alpha_x}{ \alpha_x + \beta_x} $$

# %%
# plot limiting value of v.
plt.figure()
plt.plot(V, alpha_n/(alpha_n+beta_n), 'b', label='n')
plt.plot(V, alpha_m/(alpha_m+beta_m), 'r', label='m')
plt.plot(V, alpha_h/(alpha_h+beta_h), 'g', label='h')
plt.ylabel('Stable Value')
plt.xlabel("Membrane Voltage(mV)")
plt.title('Stable Values of gate variables')
plt.legend()

# %% [markdown]
# The graph represents that the steady values of $m$ ($Na^+$ active) and $n$ ($K^+$) increase as the increment of voltage, which reflects the opening of ion channels. While the decrement of $h$ ($Na^+$ inactive) indicates the closing of $Na^+$ channels during repolarization stage.

# %% [markdown]
# ### Time constant

# %% [markdown]
# The time constant $\tau_x$ for the evolution of $x$ channel is given by
#
# $$
# \tau_x = {1 \over \alpha_x + \beta_x}
# $$

# %%
# plot time constant (tau) of v.

# compute tau
tau_m = 1/(alpha_m+beta_m)
tau_h = 1/(alpha_h+beta_h)
tau_n = 1/(alpha_n+beta_n)

# plot
plt.figure()
plt.plot(V, tau_n, 'b', label='n')
plt.plot(V, tau_m, 'r', label='m')
plt.plot(V, tau_h, 'g', label='h')
plt.ylabel('Time constant')
plt.xlabel("Membrane Voltage(mV)")
plt.title('Time constant of V')
plt.legend()


# %% [markdown]
# The time constant of $m$ is much smaller than $n$ and $h$ during resting potential, thus only $Na^+$ channel opens during the depolarization stage.
#
# The time constants of $n$ and $h$ decrease as voltage increases, which represents the closing of $Na^+$ channel and opening of $K^+$ channel.

# %% [markdown]
# ## Model Analysis

# %% [markdown]
# To see how the model respond to various input currents, let's start by defining a method to run simulation and visualize the dynamics.

# %%
def HH_simulate(input_current, duration, geometry = (1,), n_figs=2):
    '''Run simulation with HH model and visualize it.
    
    Args:
        input_current(NPArray).
        duration(float): duration of the input current.
        n_figs(int): number of figures to plot. Default 2 are Input current and membrane potential.
            3: plot gate variable.
            4: plot gate variable of V.
    '''

    if n_figs < 2:
        n_figs = 2
        print("n_figs must be >= 2!")
    

    # create HH_neuron
    HH_neuron = brainmodels.neurons.HH(geometry, monitors=['spike', 'V', 'm', 'h', 'n'])

    # set initial potential (between -55 and -75)
    HH_neuron.V[:] = bp.math.random.random(geometry) * 20 + -75

    # apply input current to the HH_neuron
    HH_neuron.run(duration=duration, inputs=("input", input_current, 'iter'))

    # read time sequence
    ts = HH_neuron.mon.ts
    fig, gs = bp.visualize.get_figure(n_figs, 1, 3, 6)
    
    V = HH_neuron.mon.V[:, 0]    
    
    # plot input current
    fig.add_subplot(gs[0, 0])
    try:
        plt.plot(ts, input_current, 'r')
    except:
        plt.plot(ts, input_current* bp.math.zeros(len(ts)), 'r')
    else:
        pass
    plt.ylabel('Input Current')
    plt.xlim(-0.1, duration + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('Input Current')
        
    # plot membrane potential
    fig.add_subplot(gs[1, 0])
    plt.plot(ts, V)
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, duration + 0.1)
    plt.ylim(-95., 60.)
    plt.title('Membrane potential')
    
    if n_figs > 2:
        # plot gate variables
        fig.add_subplot(gs[2, 0])
        plt.plot(ts, HH_neuron.mon.m[:, 0], label='Na active (m)')
        plt.plot(ts, HH_neuron.mon.h[:, 0], label='Na inactive (h)')
        plt.plot(ts, HH_neuron.mon.n[:, 0], label='K (n)')
        plt.legend()
        plt.xlim(-0.1, duration + 0.1)
        plt.xlabel('Time (ms)')
        plt.ylabel('gate variables')
        plt.title('gate variables')
        
        if n_figs > 3:
            # plot n, m, h of v.
            fig.add_subplot(gs[3, 0])
            plt.plot(V, HH_neuron.mon.n[:, 0], 'b', label='n')
            plt.plot(V, HH_neuron.mon.m[:, 0], 'r', label='m')
            plt.plot(V, HH_neuron.mon.h[:, 0], 'g', label='h')
            plt.ylabel('Gate variables')
            plt.xlabel("Membrane Voltage(mV)")
            plt.title('Gate variables of V')
            plt.legend()

# %% [markdown]
# ### Find critical current

# %% [markdown]
# Let's determine the lowest step current amplitude $I_{min}$ for generating at least one spike by trying different input amplitudes. (The duration of step current is 5 ms.)

# %%
# apply several step currents
amplitudes = [0, 1.9, 0, 1.95, 0, 2.0]  # step current amplitude

stim_interval = 25.
stim_dur = 5.   # stimulation duration
I = []

for amplitude in amplitudes:
    I.append((0., stim_interval))
    I.append((amplitude, stim_dur))
    I.append((0., stim_interval))
    
(step_I, duration) = bp.inputs.constant_current(I)

# run simulation
HH_simulate(input_current=step_I, duration=duration, geometry = (10,))

# %% [markdown]
# It seems that $I_{min}$ is between 1.95 and 2.0.

# %% [markdown]
# #### Minimal current to generate repetitive firing

# %% [markdown]
# We can use the same way to determine the lowest step current amplitude to generate repetitive firing.

# %%
# apply several step currents
amplitudes = [5., 5.5, 6.]  # step current amplitude

stim_interval = 100.
stim_dur = 20.   # stimulation duration
I = []

for amplitude in amplitudes:
    I.append((0., stim_interval))
    I.append((amplitude, stim_dur))
    I.append((0., stim_interval))
    
(step_I, duration) = bp.inputs.constant_current(I)

# run simulation
HH_simulate(input_current=step_I, duration=duration)

# %% [markdown]
# It seems that $I_{min}$ is between 5.5 and 6.

# %% [markdown]
# ### Slow ramp current response

# %% [markdown]
# Now let's inject a slow ramp current into a HH neuron and find the lowest amplitude that can elicit a spike.

# %%
# apply slow ramp current
duration=80.

ramp_I = bp.inputs.ramp_current(c_start = 0., 
                                c_end = 3.5, 
                                duration = duration,
                                t_start = 40.,
                                t_end = 60.)

# run simulation
HH_simulate(input_current=ramp_I, duration=duration, n_figs=4)

# %% [markdown]
# ### Fast ramp current response

# %% [markdown]
# Now inject a fast ramp current and find it's critical amplitude.

# %%
# apply fast ramp current
duration = 68.

ramp_I = bp.inputs.ramp_current(c_start = 0., 
                                c_end = 6.25, 
                                duration = duration,
                                t_start = 40.,
                                t_end = 48.)

# run simulation
HH_simulate(input_current=ramp_I, duration=duration, n_figs=4)

# %% [markdown]
# ## Reference

# %% [markdown]
# [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description of membrane current and its application to conduction and excitation in nerve." The Journal of physiology 117.4 (1952): 500.
