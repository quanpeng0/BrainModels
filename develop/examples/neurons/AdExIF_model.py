import brainpy as bp
import brainmodels

bp.backend.set(backend='numpy', dt=.005)

duration = 200
I_ext = 65
neu= brainmodels.neurons.AdExIF(size=1, monitors=['V', 'spike', 'refractory'],
                                   a=.5, b=7, R=.5, tau=9.9, tau_w=100,
                                   V_reset=-70, V_rest=-70, V_th=-30,
                                   V_T=-50, delta_T=2., t_refractory=5.)

neu.run(duration, inputs=('input', I_ext))
fig, gs = bp.visualize.get_figure(3, 1, 4, 10)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(neu.mon.ts, neu.mon.V,
                       xlabel="t", ylabel="V")
fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(neu.mon.ts, neu.mon.spike,
                       xlabel="t", ylabel="V")
fig.add_subplot(gs[2, 0])
bp.visualize.line_plot(neu.mon.ts, neu.mon.refractory,
                       xlabel="t", ylabel="V", show=True)
