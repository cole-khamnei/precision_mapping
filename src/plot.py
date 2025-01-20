import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from . import surface_mapping as sfm

# ----------------------------------------------------------------------------# 
# --------------------               Plots                --------------------# 
# ----------------------------------------------------------------------------# 


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


# \section precision mapping plots


def plot_precision_map(precision_map_values, title="", save_path=None):
    """ """
    n_parcels = max(np.nanmax(precision_map_values["left"]), np.nanmax(precision_map_values["right"]))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax, _ = sfm.surface_plot(precision_map_values, cmap=plt.cm.Spectral, ax=ax)
    ax.set_title(f"{title} Precision Map\nNumber of Parcels: {n_parcels:0.0f}")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)


def precision_map_QC_plots(partition, save_path=None):
    """ """
    index, groups = partition
    unique_groups, counts = np.unique(groups, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.kdeplot(counts, ax=ax, bw_method=0.1,
                label=f"Total Communities Found: {len(counts)}\nMedian Size: {np.median(counts) // 1}\nMax Size: {np.max(counts)}")
    ax.set(xlabel="Cluster Vertices", title="Distribution of Infomap Cluster Size")
    ax.legend(title="")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)


def write_dlabel_precision_map(precision_map_values, save_path, label=""):
    """ """
    precision_map_labels = precision_map_values.copy()
    precision_map_labels["left"] = precision_map_labels["left"].astype(str)
    precision_map_labels["right"] = precision_map_labels["right"].astype(str)
    sfm.write_labels_to_dlabel(precision_map_labels, save_path, label_name=label)


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
