# -*- coding: utf-8 -*-

import importlib
import inspect
import os


def write(module_name, filename, header=None):
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

  module = importlib.import_module(module_name)
  models = [k for k in dir(module)
            if not k.startswith('__') and not inspect.ismodule(getattr(module, k))]

  fout = open(filename, 'w')

  # write header
  if header is None:
    header = f'``{module_name}`` module'
  else:
    header = header
  fout.write(header + '\n')
  fout.write('=' * len(header) + '\n\n')
  fout.write(f'.. currentmodule:: {module_name} \n')
  fout.write(f'.. automodule:: {module_name} \n\n')

  # write autosummary
  fout.write('.. autosummary::\n')
  fout.write('   :toctree: generated/\n\n')
  for m in models:
    fout.write(f'   {m}\n')

  # write autoclass
  fout.write('\n')
  for m in models:
    fout.write(f'.. autoclass:: {m}\n')
    fout.write(f'   :members:\n\n')

  fout.close()


def generate():
  write(module_name='brainmodels.neurons', filename='apis/neurons.rst')
  write(module_name='brainmodels.synapses', filename='apis/synapses.rst')
  write(module_name='brainmodels.channels.Na_channels', filename='apis/channels.Na.rst', header='Na')
  write(module_name='brainmodels.channels.K', filename='apis/channels.K.rst', header='K')
  write(module_name='brainmodels.channels.Ca_channels_channels', filename='apis/channels.Ca.rst', header='Ca')
  write(module_name='brainmodels.channels.IH_channels', filename='apis/channels.IH.rst', header='IH')
  write(module_name='brainmodels.channels.KCa_channels', filename='apis/channels.KCa.rst', header='KCa')
  write(module_name='brainmodels.channels.other_channels', filename='apis/channels.other.rst', header='Other')

