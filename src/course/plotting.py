"""Shared plot style for the course notebooks (warm, restrained)."""
import matplotlib.pyplot as plt

INK = "#33281f"
OCHRE = "#b06a12"
TEAL = "#2f5d62"
SAND = "#cdb89d"
ROSE = "#9c4a3c"
PALETTE = [OCHRE, TEAL, ROSE, SAND, INK]


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.prop_cycle": plt.cycler(color=PALETTE), "legend.frameon": False,
        "font.sans-serif": ["Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei",
                            "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
