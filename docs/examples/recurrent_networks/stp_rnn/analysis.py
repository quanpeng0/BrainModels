# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
  'get_perf',
]


def get_perf(target, output, mask):
  """Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on, e.g. when y[:,:,0] = 0 """
  target = target.numpy()
  output = output.numpy()
  mask = mask.numpy()

  mask_full = mask > 0
  mask_test = mask_full * (target[:, :, 0] == 0)
  mask_non_match = mask_full * (target[:, :, 1] == 1)
  mask_match = mask_full * (target[:, :, 2] == 1)
  target_max = np.argmax(target, axis=2)
  output_max = np.argmax(output, axis=2)

  match = target_max == output_max
  accuracy = np.sum(match * mask_test) / np.sum(mask_test)
  acc_non_match = np.sum(match * np.squeeze(mask_non_match)) / np.sum(mask_non_match)
  acc_match = np.sum(match * np.squeeze(mask_match)) / np.sum(mask_match)
  return accuracy, acc_non_match, acc_match