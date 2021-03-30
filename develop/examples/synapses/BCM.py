import brainpy as bp
import brainmodels
import numpy as np
import matplotlib.pyplot as plt

bp.backend.set(backend='numpy', dt=.02)

class neu(bp.NeuGroup):
    target_backend = 'general'

    def __init__(self, size, tau=10., **kwargs):
        self.tau = tau
        
        self.r = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)

        super(neu, self).__init__(size=size, **kwargs)

    @staticmethod
    @bp.odeint
    def g(r, t, I, tau):
        return -r / tau + I

    def update(self, _t):
        self.r = self.g(self.r, _t, self.input, self.tau)
        self.input[:] = 0


w_max = 2.
n_post = 1
n_pre = 20

# group selection
group1, duration = bp.inputs.constant_current(([1.5, 1], [0, 1])*20)
group2, duration = bp.inputs.constant_current(([0, 1], [1., 1])*20)

input_r = np.vstack((
                    (group1,)*10, (group2,)*10
                    ))

pre = neu(n_pre, monitors=['r'])
post = neu(n_post, monitors=['r'])
bcm = brainmodels.synapses.BCM(pre=pre, post=post, 
                    conn=bp.connect.All2All(),
                    monitors=['w'])

net=bp.Network(pre, bcm, post)
net.run(duration, inputs=(pre, 'r', input_r.T, "="))

w1 = np.mean(bcm.mon.w[:,:10,0], 1)
w2 = np.mean(bcm.mon.w[:,10:,0], 1)

r1 = np.mean(pre.mon.r[:, :10], 1)
r2 = np.mean(pre.mon.r[:, 10:], 1)
post_r = np.mean(post.mon.r[:, :], 1)

fig, gs = bp.visualize.get_figure(3, 1, 2, 6)
fig.add_subplot(gs[2, 0], xlim=(0, duration), ylim=(0, w_max))
plt.plot(net.ts, w1, 'b', label='group1')
plt.plot(net.ts, w2, 'r', label='group2')
plt.title("weights")
plt.ylabel("weights")
plt.xlabel("t")
plt.legend()

fig.add_subplot(gs[0, 0], xlim=(0, duration))
plt.plot(net.ts, r1, 'b', label='group1')
plt.plot(net.ts, r2, 'r', label='group2')
plt.title("inputs")
plt.ylabel("firing rate")
plt.xlabel("t")
plt.legend()

fig.add_subplot(gs[1, 0], xlim=(0, duration))
plt.plot(net.ts, post_r, 'g', label='post_r')
plt.title("response")
plt.ylabel("firing rate")
plt.xlabel("t")
plt.legend()

plt.show()

