import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# \section plots


def evar_timeseries_plot(ev_s, axes=None, label=""):
    """ """

    if axes is None:
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [8, 1]})
        fig.tight_layout(w_pad=-2)
    else:
        a0, a1 = axes

    if ev_s.ndim > 1:
        n_trs = len(tr_s)
        tr_s = np.arange(n_trs)
        alpha = 1.96
        m, sd = np.mean(ev_s, axis=0), np.std(ev_s, axis=0)
        a0.plot(tr_s, m, label=f"{label} Mean Subject")
        a0.fill_between(tr_s, m - alpha * sd, m + alpha * sd, alpha=0.1, label=f"{label} 95% CI")
        a0.axhline(np.mean(ev_s), linestyle="--", alpha=0.5)

    else:
        n_trs = len(ev_s)
        a0.plot(ev_s, label=label)
        tr_s = np.arange(n_trs)
        a0.fill_between(tr_s, 0, ev_s, alpha=0.1)

    a0.legend()
    a0.set_ylim(None, 1.05)
    a0.set(xlabel="TR", ylabel="Explained Variance")
    a0.set_xlim(-10, n_trs + 5)

    a1.yaxis.tick_right()
    a1.yaxis.set_label_position("right")
    sns.kdeplot(y=ev_s.ravel(), ax=a1, fill=True, bw_method=0.05, label=label)
    a1.axhline(np.mean(ev_s), linestyle="--", alpha=0.5)
    a1.set_ylim(a0.get_ylim())
    a1.set(ylabel="Explained Variance")
    a1.legend(fontsize=7)

    return (a0, a1)


# \section end
