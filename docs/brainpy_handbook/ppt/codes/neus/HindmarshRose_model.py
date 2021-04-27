class HindmarshRose(bp.NeuGroup):
  target_backend = 'general'

  @staticmethod
  def derivative(V, y, z, t, a, b, I_ext, c, d, r, s, V_rest):
    dVdt = y - a * V * V * V + b * V * V - z + I_ext
    dydt = c - d * V * V - y
    dzdt = r * (s * (V - V_rest) - z)
    return dVdt, dydt, dzdt

  def __init__(self, size, a=1., b=3.,
               c=1., d=5., r=0.01, s=4.,
               V_rest=-1.6, **kwargs):
    # parameters
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.r = r
    self.s = s
    self.V_rest = V_rest

    # variables
    num = bp.size2len(size)
    self.z = bp.ops.zeros(num)
    self.input = bp.ops.zeros(num)
    self.V = bp.ops.ones(num) * -1.6
    self.y = bp.ops.ones(num) * -10.
    self.spike = bp.ops.zeros(num, dtype=bool)

    self.integral = bp.odeint(f=self.derivative)
    super(HindmarshRose, self).__init__(size=size, **kwargs)

  def update(self, _t):
    for i in prange(self.num):
      V, self.y[i], self.z[i] = self.integral(
        self.V[i], self.y[i], self.z[i], _t,
        self.a, self.b, self.input[i],
        self.c, self.d, self.r, self.s,
        self.V_rest)
      self.V[i] = V
      self.input[i] = 0.
