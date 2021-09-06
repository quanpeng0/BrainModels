# -*- coding: utf-8 -*-

__all__ = [
  'CUBA', 'COBA'
]


class CUBA(object):
  def __init__(self):
    super(CUBA, self).__init__()

  def output_current(self, g):
    return g


class COBA(object):
  def __init__(self, post, E):
    super(COBA, self).__init__()

    self.E = E
    self.post = post
    assert hasattr(post, 'V'), 'Post-synaptic group must has "V" variable.'

  def output_current(self, g):
    return g * (self.E - self.post.V)

  def output_current_idx(self, g, i):
    return g * (self.E - self.post.V[i])
