# -*- coding: utf-8 -*-

"""
Implementation of the paper:

- Si Wu, Kosuke Hamaguchi, and Shun-ichi Amari. "Dynamics and computation
  of continuous attractors." Neural computation 20.4 (2008): 994-1025.

The mathematical equation of the Continuous-attractor Neural Network (CANN) is
given by:

\tau \frac{du(x,t)}{dt} = -u(x,t) + \rho \int dx' J(x,x') r(x',t)+I_{ext} \\

r(x,t) = \frac{u(x,t)^2}{1 + k \rho \int dx' u(x',t)^2} \\

J(x,x') = \frac{1}{\sqrt{2\pi}a}\exp(-\frac{|x-x'|^2}{2a^2}) \\

I_{ext} = A\exp\left[-\frac{|x-z(t)|^2}{4a^2}\right]

"""

import brainpy as bp
import numpy as np

bp.backend.set(backend='numpy', dt=0.05)

class CANN(bp.NeuGroup):
    target_backend = 'general'

    def __init__(self, size, rho, dx, tau=1., k=8.1,
                 **kwargs):
        # parameters
        self.rho = rho
        self.dx = dx
        self.tau = tau
        self.k = k  # Degree of the rescaled inhibition

        # variables
        self.x = bp.backend.zeros(size)
        self.u = bp.backend.zeros(size)
        self.r = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)
        self.Jxx = bp.backend.zeros((size, size))

        super(CANN, self).__init__(size=size, **kwargs)


    @staticmethod
    @bp.odeint(method='rk4')
    def integral(u, t, J_xx, I_ext, k, rho, dx, tau):
        r_num = np.square(u)
        r_den = 1.0 + k * rho * bp.backend.sum(r_num) * dx
        r = r_num / r_den
        I_rec = rho * np.dot(J_xx, r) * dx
        dudt = (-u + I_rec + I_ext) / tau
        return dudt

    def update(self, _t):
        self.u = self.integral(self.u, _t, self.Jxx, self.input, 
                                self.k, self.rho, self.dx, self.tau)
        self.input[:] = 0

# connection #
# ---------- #

def dist(d, z_range):
    d = np.remainder(d, z_range)
    d = np.where(d > 0.5 * z_range, d - z_range, d)
    return d

def make_conn(x, J0, z_range):
    assert np.ndim(x) == 1
    x_left = np.reshape(x, (len(x), 1))
    x_right = np.repeat(x.reshape((1, -1)), len(x), axis=0)
    d = dist(x_left - x_right, z_range)
    jxx = J0 * np.exp(-0.5 * np.square(d / a)) / (np.sqrt(2 * np.pi) * a)
    return jxx

if __name__ == "__main__":
    N = 256.
    
    a = 0.5  # Half-width of the range of excitatory connections
    A = 10.   # Magnitude of the external input
    
    J0 = 4. / (N / 128)
    z_min = -np.pi
    z_max = np.pi
    z_range = z_max - z_min
    rho = N / z_range  # The neural density
    dx = z_range / N

    group = CANN(size=int(N), rho=rho, dx=dx, monitors=['u'])
    group.x = np.linspace(z_min, z_max, int(N))
    group.Jxx = make_conn(group.x, J0, z_range)


    # population coding
    I1 = A * np.exp(-0.25 * np.square(dist(group.x - 0., z_range) / a))
    Iext, duration = bp.inputs.constant_current([(0., 1.), (I1, 8.), (0., 8.)])
    group.run(duration=duration, inputs=('input', Iext))

    bp.visualize.animate_1D(
        dynamical_vars=[{'ys': group.mon.u, 'xs': group.x, 'legend': 'u'},
                        {'ys': Iext, 'xs': group.x, 'legend': 'Iext'}],
        show=True,
        frame_step=1,
        frame_delay=100,
        # save_path='encoding.gif'
    )

    # template matching
    dur1, dur2, dur3 = 10., 30., 0.
    num1 = int(dur1 / bp.backend._dt)
    num2 = int(dur2 / bp.backend._dt)
    num3 = int(dur3 / bp.backend._dt)
    Iext = np.zeros((num1 + num2 + num3, group.size[0]))
    Iext[:num1] = A * np.exp(-0.25 * np.square(dist(group.x + 0.5, z_range) / a))
    Iext[num1:num1 + num2] = A * np.exp(-0.25 * np.square(dist(group.x - 0., z_range) / a))
    Iext[num1:num1 + num2] += 0.1 * A * np.random.randn(num2, group.size[0])
    group.run(duration=dur1 + dur2 + dur3, inputs=('input', Iext))

    bp.visualize.animate_1D(
        dynamical_vars=[{'ys': group.mon.u, 'xs': group.x, 'legend': 'u'},
                        {'ys': Iext, 'xs': group.x, 'legend': 'Iext'}],
        show=True,
        frame_step=5,
        frame_delay=50,
        # save_path='decoding.gif'
    )

    # smooth tracking
    dur1, dur2, dur3 = 20., 20., 20.
    num1 = int(dur1 / bp.backend._dt)
    num2 = int(dur2 / bp.backend._dt)
    num3 = int(dur3 / bp.backend._dt)
    position = np.zeros(num1 + num2 + num3)
    position[num1: num1 + num2] = np.linspace(0., 12., num2)
    position[num1 + num2:] = 12.
    position = position.reshape((-1, 1))
    Iext = A * np.exp(-0.25 * np.square(dist(group.x - position, z_range) / a))
    group.run(duration=dur1 + dur2 + dur3, inputs=('input', Iext))

    bp.visualize.animate_1D(
        dynamical_vars=[{'ys': group.mon.u, 'xs': group.x, 'legend': 'u'},
                        {'ys': Iext, 'xs': group.x, 'legend': 'Iext'}],
        show=True,
        frame_step=5,
        frame_delay=50,
        # save_path='tracking.gif'
    )

