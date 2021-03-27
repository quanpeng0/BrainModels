import numpy as np
import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt

print(bp.__version__)

def get_LIF(V_rest=0., V_reset=-5., V_th=20., R=1.,
            tau=10., t_refractory=5., noise=0., mode='scalar'):
    ST = bp.types.NeuState(
        {'V': 0, 'input': 0, 'spike': 0, 'refractory': 0, 't_last_spike': -1e7}
    )

    @bp.integrate
    def int_V(V, t, I_ext):  # integrate u(t)
        return (- (V - V_rest) + R * I_ext) / tau, noise / tau

    if mode == 'scalar':
        def update(ST, _t):
            # update variables
            ST['spike'] = 0.
            if _t - ST['t_last_spike'] <= t_refractory:
                ST['refractory'] = 1.
            else:
                ST['refractory'] = 0.
                V = int_V(ST['V'], _t, ST['input'])
                if V >= V_th:
                    V = V_reset
                    ST['spike'] = 1
                    ST['t_last_spike'] = _t
                ST['V'] = V
        
        def reset(ST):
            ST['input'] = 0.  # reset input here or it will be brought to next step

    return bp.NeuType(name='LIF_neuron',
                      ST=ST,
                      steps=[update, reset],
                      mode=mode)

LIF = get_LIF(V_rest=-65., V_reset=-65., V_th=-55.)

neu = bp.NeuGroup(LIF, 1, monitors=['input'])
neu.ST['V'] = -65.
neu.runner.set_schedule(['input', 'update', 'monitor', 'reset'])

duration=500
I = bp.inputs.spike_current([0.1], bp.profile._dt, 1., duration=duration)

# NMDA
syn_model = brainmodels.synapses.get_NMDA(g_max=0.03)
syn = bp.SynConn(syn_model, pre_group=neu, post_group=neu, conn=bp.connect.All2All())
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(syn, 'pre.spike', I, '='))
plt.plot(net.ts, 5000 * neu.mon.input[:, 0], label='NMDA')

# AMPA
syn_model = brainmodels.synapses.get_AMPA1(g_max=0.001)
syn = bp.SynConn(syn_model, pre_group=neu, post_group=neu, conn=bp.connect.All2All())
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(syn, 'pre.spike', I, '='))
plt.plot(net.ts, 5000 * neu.mon.input[:, 0], label='AMPA')

# GABA_b
syn_model = brainmodels.synapses.get_GABAb1(T_duration=0.15)
syn = bp.SynConn(syn_model, pre_group=neu, post_group=neu, conn=bp.connect.All2All())
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(syn, 'pre.spike', I, '='))
plt.plot(net.ts, 1e+6 * 5000 * neu.mon.input[:, 0], label='GABAb')

# GABA_a
syn_model = brainmodels.synapses.get_GABAa1(g_max=0.002)
syn = bp.SynConn(syn_model, pre_group=neu, post_group=neu, conn=bp.connect.All2All())
net = bp.Network(neu, syn)
net.run(duration=duration, inputs=(syn, 'pre.spike', I, '='))
plt.plot(net.ts, 5000 * neu.mon.input[:, 0], label='GABAa')


plt.ylabel('-I')
plt.xlabel('Time (ms)')
plt.legend()
plt.show()