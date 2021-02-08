
"""
“Fast Global Oscillations in Networks of Integrate-and-Fire Neurons with Low Firing Rates”
Nicolas Brunel & Vincent Hakim Neural Computation 11, 1621-1671 (1999)
"""


import brainpy as bp
import numpy as np

bp.profile.set(jit=True)

Vr = 10.  # mV
theta = 20.  # mV
tau = 20.  # ms
delta = 2.  # ms
taurefr = 2.  # ms
duration = 100.  # ms
J = .1  # mV
muext = 25.  # mV
sigmaext = 1.  # mV
C = 1000
N = 5000
sparseness = float(C) / N


@bp.integrate
def int_v(V, t):
    return (-V + muext) / tau, sigmaext / np.sqrt(tau)


def neu_update(ST, _t):
    ST['spike'] = 0.
    ST['not_ref'] = 0.
    if (_t - ST['t_last_spike']) > taurefr:
        V = int_v(ST['V'], _t)
        if V > theta:
            ST['spike'] = 1.
            ST['V'] = Vr
            ST['t_last_spike'] = _t
        else:
            ST['V'] = V
            ST['not_ref'] = 1.


lif = bp.NeuType(name='lif',
                 ST=bp.NeuState('spike', V=Vr, t_last_spike=-1e7, not_ref=1.),
                 steps=neu_update,
                 mode='scalar')


def syn_update(ST, pre):
    if pre['spike']:
        ST['g'] = J
    else:
        ST['g'] = 0.


@bp.delayed
def syn_output(ST, post):
    if post['not_ref']:
        post['V'] -= ST['g']


syn = bp.SynType(name='syn', ST=bp.SynState(['g']),
                 steps=(syn_update, syn_output),
                 mode='scalar')

group = bp.NeuGroup(lif, geometry=N, monitors=['spike', 'V'])
conn = bp.SynConn(syn, pre_group=group, post_group=group,
                  conn=bp.connect.FixedProb(sparseness),
                  delay=delta)
net = bp.Network(group, conn)
net.run(duration, report=True)

bp.visualize.raster_plot(net.ts, group.mon.spike, xlim=(0, duration), show=True)
