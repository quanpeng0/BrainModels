# -*- coding: utf-8 -*-


import os
import importlib


def write(module_name, filename):
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

  module = importlib.import_module(module_name)
  models = [k for k in dir(module) if not k.startswith('__')]

  fout = open(filename, 'w')

  # write header
  fout.write(module_name + '\n')
  fout.write('=' * len(module_name) + '\n\n')
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

