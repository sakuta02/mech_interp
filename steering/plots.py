"""Графики опыта: доля примеров, где модель выдала целевое значение, по слоям и по силе вмешательства,
для каждого инструмента и перехода; отдельный рисунок для переходов 3→5 и 5→3 со своим и неродным вектором."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import TOOLS, TRANSITIONS

MAIN = ["steer_tool", "steer_foreign"]
CTRL = ["steer_neg", "shuffled", "random"]
COL = dict(steer_tool="C0", steer_foreign="C3", steer_neg="C4", shuffled="C7", random="C8", patch="k")


def grid_plot(summary, out_dir, cells, kinds, xcol, mask, xlabel, fname, title):
    """Сетка графиков: cells[i][j] = (id инструмента или ALL, переход o_d); в каждой ячейке по линии на вид условия."""
    sm = summary[mask]
    fig, axes = plt.subplots(len(cells), len(cells[0]), figsize=(3.4 * len(cells[0]) + 1, 2.5 * len(cells) + 1.2),
                             sharey=True, squeeze=False)
    for i, row in enumerate(cells):
        for j, (tool, tr) in enumerate(row):
            ax = axes[i, j]
            for kind in kinds:
                q = sm[(sm.kind == kind) & (sm.tool_id == tool) & (sm.trans == tr)].dropna(subset=[xcol]).sort_values(xcol)
                if len(q):
                    ax.plot(q[xcol], q.pT, color=COL[kind], ls="--" if kind in CTRL + ["patch"] else "-",
                            marker="o", ms=3, label=kind)
            ax.set_title(f"{tool} | {tr.replace('_', '→')}", fontsize=8); ax.grid(alpha=0.3)
            if i == len(cells) - 1: ax.set_xlabel(xlabel)
            if j == 0: ax.set_ylabel("доля целевого значения")
    axes[0, 0].legend(fontsize=6)
    plt.suptitle(title); plt.tight_layout()
    plt.savefig(f"{out_dir}/{fname}", dpi=130); plt.close()


def make_plots(cfg, summary, log):
    """Три рисунка: все переходы по слоям, все переходы по силе, переходы 3→5 и 5→3 со своим и неродным вектором."""
    L0, A0 = cfg.L0, cfg.A0
    tools = [t["id"] for t in TOOLS] + ["ALL"]
    trans = [f"{o}_{d}" for o, d in TRANSITIONS]
    kinds = MAIN + CTRL + ["patch"]
    by_tool = [[(tl, tr) for tr in trans] for tl in tools]
    foreign_focus = [[(tl, tr) for tl in tools] for tr in ("3_5", "5_3")]
    on_layer = (summary.alpha == A0) | (summary.kind == "patch")
    grid_plot(summary, cfg.OUT_DIR, by_tool, kinds, "layer", on_layer, "слой", f"pT_vs_layer_A{A0}.png",
              f"доля целевого значения по слоям, alpha {A0}")
    grid_plot(summary, cfg.OUT_DIR, by_tool, kinds, "alpha", summary.layer == L0, "α", f"pT_vs_alpha_L{L0}.png",
              f"доля целевого значения по силе, слой {L0}")
    grid_plot(summary, cfg.OUT_DIR, foreign_focus, MAIN + ["shuffled", "patch"], "layer", on_layer, "слой",
              f"foreign_3to5_5to3_vs_layer_A{A0}.png",
              f"переходы 3→5 и 5→3: свой вектор (steer_tool) и вектор неродных инструментов (steer_foreign), alpha {A0}")
    log("графики сохранены")
