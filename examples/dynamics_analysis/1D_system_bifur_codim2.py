# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import brainpy as bp


def define_model():
    """
    A 1D model for codimension 2 bifurcation testing.

    .. math::

        \dot{x} = \mu+ \lambda x - x**3
    """

    lambd = 0
    mu = 0

    @bp.integrate
    def int_x(x, t):
        dxdt = mu + lambd * x - x ** 3
        return dxdt

    def update(ST, _t):
        ST['x'] = int_x(ST['x'], _t)

    return bp.NeuType(name="dummy_model",
                      ST=bp.types.NeuState({'x': 0.}),
                      steps=update)


analyzer = bp.analysis.Bifurcation(
    model=define_model(),
    target_pars={'mu': [-4, 4], 'lambd': [-1, 4]},
    target_vars={'x': [-3, 3]},
    numerical_resolution=0.1)
analyzer.plot_bifurcation(show=True)
