import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import matplotlib as mpl, matplotlib.pyplot as plt, string
PAL = dict(blue="#2B6CB0", orange="#E07B00", green="#2E8B57", red="#C0392B", purple="#7B4FA3", grey="#6B7280", teal="#1B998B", gold="#C9A227", light="#E5E7EB")
CYC = [PAL[k] for k in ["blue", "orange", "green", "red", "purple", "teal", "gold", "grey"]]
def setup():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "axes.prop_cycle": mpl.cycler(color=CYC),
        "figure.dpi": 110, "savefig.dpi": 600, "savefig.bbox": "tight", "axes.grid": False, "lines.linewidth": 1.4, "pdf.fonttype": 42, "svg.fonttype": "none"})
def panel(axs, size=11):
    for a, l in zip(axs, string.ascii_lowercase): a.text(-0.12, 1.06, f"({l})", transform=a.transAxes, fontsize=size, fontweight="bold", va="bottom")
def save(fig, name):
    d = ROOT + "analysis/figures/"
    fig.savefig(d + name + ".png"); fig.savefig(d + name + ".pdf"); plt.close(fig)
