# -*- coding: utf-8 -*-

# %%
import sys
sys.path.append('/mnt/d/codes/Projects/BrainPy/')
from functools import partial

import brainpy as bp
import brainpy.math.jax as bm

bp.math.use_backend('jax')

bp.__version__

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# %%
# Integration parameters
T = 1.0  # Arbitrary amount time, roughly physiological.
dt = 0.04
num_step = int(T / dt)  # Divide T into this many bins
bval = 0.01  # bias value limit
sval = 0.025  # standard deviation (before dividing by sqrt(dt))

# %%
# The scaling of the recurrent parameters in an RNN really matters.
# The correct scaling is 1/sqrt(number of recurrent inputs), which
# yields an order 1 signal output to a neuron if the input is order 1.
# Given that VRNN uses a tanh nonlinearity, with min and max output
# values of -1 and 1, this works out.  The scaling just below 1
# (0.95) is because we know we are making a line attractor so, we
# might as well start it off basically right 1.0 is also basically
# right, but perhaps will lead to crazier dynamics.
param_scale = 0.85  # Scaling of the recurrent weight matrix

# %%
# Optimization hyperparameters
l2reg = 0.0002  # amount of L2 regularization on the weights
num_train = 10000  # Total number of batches to train on.
num_batch = 128  # How many examples in each batch
max_grad_norm = 5.0  # Gradient clipping is HUGELY important for training RNNs
# max gradient norm before clipping, clip to this value.


# %%
def plot_examples(num_time, inputs, hiddens, outputs, targets, num_example=1, num_plot=10):
  """Plot some input/hidden/output triplets."""
  plt.figure(figsize=(num_example * 5, 14))

  for bidx in range(num_example):
    plt.subplot(3, num_example, bidx + 1)
    plt.plot(inputs[bidx, :], 'k')
    plt.xlim([0, num_time])
    plt.title('Example %d' % bidx)
    if bidx == 0: plt.ylabel('Input Units')

  closeness = 0.25
  for bidx in range(num_example):
    plt.subplot(3, num_example, num_example + bidx + 1)
    plt.plot(hiddens[bidx, :, 0:num_plot] + closeness * np.arange(num_plot), 'b')
    plt.xlim([0, num_time])
    if bidx == 0: plt.ylabel('Hidden Units')

  for bidx in range(num_example):
    plt.subplot(3, num_example, 2 * num_example + bidx + 1)
    plt.plot(outputs[bidx, :, :], 'r', label='predict')
    plt.plot(targets[bidx, :, :], 'k', label='target')
    plt.xlim([0, num_time])
    plt.xlabel('Time steps')
    plt.legend()
    if bidx == 0: plt.ylabel('Output Units')

  plt.show()


# %%
def plot_params(rnn):
  """Plot the parameters. """
  assert isinstance(rnn, GRU)

  plt.figure(figsize=(16, 8))
  plt.subplot(231)
  plt.stem(rnn.w_ro.numpy()[:, 0])
  plt.title('W_ro - output weights')

  plt.subplot(232)
  plt.stem(rnn.h0)
  plt.title('h0 - initial hidden state')

  plt.subplot(233)
  plt.imshow(rnn.w_rr.numpy(), interpolation=None)
  plt.colorbar()
  plt.title('W_rr - recurrent weights')

  plt.subplot(234)
  plt.stem(rnn.w_ir.numpy()[0, :])
  plt.title('W_ir - input weights')

  plt.subplot(235)
  plt.stem(rnn.b_rr.numpy())
  plt.title('b_rr - recurrent biases')

  plt.subplot(236)
  evals, _ = np.linalg.eig(rnn.w_rr.numpy())
  x = np.linspace(-1, 1, 1000)
  plt.plot(x, np.sqrt(1 - x ** 2), 'k')
  plt.plot(x, -np.sqrt(1 - x ** 2), 'k')
  plt.plot(np.real(evals), np.imag(evals), '.')
  plt.axis('equal')
  plt.title('Eigenvalues of W_rr')

  plt.show()


# %%
def plot_data(num_time, inputs, targets=None, outputs=None, errors=None, num_plot=10):
  """Plot some white noise / integrated white noise examples.

  Parameters
  ----------
  num_time : int
  num_plot : int
  inputs: ndarray
    with the shape of (num_batch, num_time, num_input)
  targets: ndarray
    with the shape of (num_batch, num_time, num_output)
  outputs: ndarray
    with the shape of (num_batch, num_time, num_output)
  errors: ndarray
    with the shape of (num_batch, num_time, num_output)
  """
  num = 1
  if errors is not None: num += 1
  if (targets is not None) or (outputs is not None): num += 1
  plt.figure(figsize=(14, 4 * num))

  # inputs
  plt.subplot(num, 1, 1)
  plt.plot(inputs[:, 0:num_plot, 0])
  plt.xlim([0, num_time])
  plt.ylabel('Noise')

  legends = []
  if outputs is not None:
    plt.subplot(num, 1, 2)
    plt.plot(outputs[:, 0:num_plot, 0])
    plt.xlim([0, num_time])
    legends.append(mlines.Line2D([], [], color='k', linestyle='-', label='predict'))
  if targets is not None:
    plt.subplot(num, 1, 2)
    plt.plot(targets[:, 0:num_plot, 0], '--')
    plt.xlim([0, num_time])
    plt.ylabel("Integration")
    legends.append(mlines.Line2D([], [], color='k', linestyle='--', label='target'))
  if len(legends): plt.legend(handles=legends)

  if errors is not None:
    plt.subplot(num, 1, 3)
    plt.plot(errors[:, 0:num_plot, 0], '--')
    plt.xlim([0, num_time])
    plt.ylabel("|Errors|")

  plt.xlabel('Time steps')
  plt.show()


# %%
@partial(bm.jit, vars=bp.ArrayCollector({'a': bm.random.DEFAULT}), static_argnums=(2, 3, 4))
def build_inputs_and_targets(mean, scale):
  """Build white noise input and integration targets."""

  # Create the white noise input.
  sample = bm.random.normal(size=(num_batch,))
  bias = mean * 2.0 * (sample - 0.5)
  samples = bm.random.normal(size=(num_step, num_batch))
  noise_t = scale / dt ** 0.5 * samples
  white_noise_t = bias + noise_t
  inputs_txbx1 = bm.expand_dims(white_noise_t, axis=2)

  # * dt, intentionally left off to get output scaling in O(1).
  integration_txbx1 = bm.expand_dims(bm.cumsum(white_noise_t, axis=0), axis=2)
  targets_txbx1 = bm.zeros_like(integration_txbx1)
  targets_txbx1[-1] = 2.0 * ((integration_txbx1[-1] > 0.0) - 0.5)
  targets_mask = bm.ones((num_batch, 1)) * (num_step - 1)
  return inputs_txbx1, targets_txbx1, targets_mask


# %%
# Plot the example inputs and targets for the RNN.
_ints, _outs, _ = build_inputs_and_targets(bval, sval)

plot_data(num_step, inputs=_ints, targets=_outs)


# %%
class GRU(bp.DynamicalSystem):
  def __init__(self, num_hidden, num_input, num_output, num_batch, l2_reg=0., **kwargs):
    super(GRU, self).__init__(num_hidden, num_input, **kwargs)

    # parameters
    self.l2_reg = l2_reg
    self.num_input = num_input
    self.num_batch = num_batch
    self.num_hidden = num_hidden
    self.num_output = num_output
    self.rng = bm.random.RandomState()

    # recurrent weights
    self.w_iz = bm.TrainVar(self.rng.normal(scale=1 / num_input ** 0.5, size=(num_input, num_hidden)))
    self.w_ir = bm.TrainVar(self.rng.normal(scale=1 / num_input ** 0.5, size=(num_input, num_hidden)))
    self.w_ia = bm.TrainVar(self.rng.normal(scale=1 / num_input ** 0.5, size=(num_input, num_hidden)))
    self.w_hz = bm.TrainVar(self.rng.normal(scale=1 / num_hidden ** 0.5, size=(num_hidden, num_hidden)))
    self.w_hr = bm.TrainVar(self.rng.normal(scale=1 / num_hidden ** 0.5, size=(num_hidden, num_hidden)))
    self.w_ha = bm.TrainVar(self.rng.normal(scale=1 / num_hidden ** 0.5, size=(num_hidden, num_hidden)))
    self.bz = bm.TrainVar(bm.zeros((num_hidden,)))
    self.br = bm.TrainVar(bm.zeros((num_hidden,)))
    self.ba = bm.TrainVar(bm.zeros((num_hidden,)))
    self.h0 = bm.TrainVar(self.rng.normal(scale=0.1, size=(num_hidden,)))

    # output weights
    self.w_ro = bm.TrainVar(self.rng.normal(scale=1 / num_hidden ** 0.5, size=(num_hidden, num_output)))
    self.b_ro = bm.TrainVar(bm.zeros((num_output,)))

    # variables
    self.h = bm.Variable(self.rng.normal(scale=0.1, size=(num_batch, self.num_hidden)))
    self.o = bm.Variable(self.h @ self.w_ro)

    # loss
    self.total_loss = bm.Variable(bm.zeros(1))
    self.l2_loss = bm.Variable(bm.zeros(1))
    self.mse_loss = bm.Variable(bm.zeros(1))

  def update(self, x, **kwargs):
    z = bm.sigmoid(x @ self.w_iz + self.h @ self.w_hz + self.bz)
    r = bm.sigmoid(x @ self.w_ir + self.h @ self.w_hr + self.br)
    a = bm.tanh(x @ self.w_ia + (r * self.h) @ self.w_ha + self.ba)
    self.h.value = (1 - z) * self.h + z * a
    self.o.value = self.h @ self.w_ro + self.b_ro

  def predict(self, xs):
    self.h[:] = self.h0
    f = bm.easy_scan(self.update, dyn_vars=self.vars().unique(), out_vars=[self.h, self.o])
    return f(xs)

  def loss(self, xs, ys):
    hs, os = self.predict(xs)
    l2 = self.l2_reg * bm.losses.l2_norm([self.w_ir, self.w_rr, self.b_rr,
                                          self.w_ro, self.b_ro, self.h]) ** 2
    mse = bm.losses.mean_squared_error(os, ys)
    total = l2 + mse
    self.total_loss[0] = total
    self.l2_loss[0] = l2
    self.mse_loss[0] = mse
    return total

