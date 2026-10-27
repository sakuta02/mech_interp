"""Графики: доля примеров, где модель выдала целевое значение, по силе вмешательства и по слоям."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAM = ["steer", "steer_same", "steer_opp", "steer_layout", "steer_dom", "steer_tool", "steer_pos"]
CTRL = ["steer_neg", "shuffled", "random"]
COL = dict(steer="C0", steer_same="C2", steer_opp="C3", steer_layout="C6", steer_dom="C9",
           steer_tool="C5", steer_pos="C1", steer_neg="C4", shuffled="C7", random="C8")


def panel_plot(summary, out_dir, dom, xcol, mask, xlabel, fname):
    """Панели графиков: строки — слот, столбцы — порядок ключей и вариант вызова, линии — условия."""
    sm = summary[(summary.domain == dom) & mask]
    cols = sorted(sm[["layout", "order"]].drop_duplicates().itertuples(index=False))
    if not cols:
        return
    fig, axes = plt.subplots(2, len(cols), figsize=(3.6 * len(cols) + 1, 7), sharey=True, squeeze=False)
    for i, slot in enumerate(["first", "second"]):
        for j, (lay, order) in enumerate(cols):
            ax = axes[i, j]
            for cnd in FAM + CTRL:
                q = sm[(sm.cond == cnd) & (sm.slot == slot) & (sm.layout == lay) & (sm.order == order)].sort_values(xcol)
                if len(q):
                    ax.plot(q[xcol], q.pT, color=COL[cnd], ls="--" if cnd in CTRL else "-", marker="o", ms=3, label=cnd)
            ax.set_title(f"{slot} | {order} | {lay}", fontsize=9); ax.grid(alpha=0.3)
            if i == 1: ax.set_xlabel(xlabel)
            if j == 0: ax.set_ylabel("доля выдачи целевого значения")
    axes[0, 0].legend(fontsize=7)
    plt.suptitle(f"тип задач: {dom}"); plt.tight_layout()
    plt.savefig(f"{out_dir}/{fname}", dpi=130); plt.close()


def make_plots(cfg, summary, log):
    """Графики по силе и по слою для каждого типа задач."""
    L0, A0 = cfg.L0, cfg.A0
    for dom in ["flight", "message", "period"]:
        panel_plot(summary, cfg.OUT_DIR, dom, "alpha", summary.layer == L0, "α", f"pT_vs_alpha_{dom}_L{L0}.png")
        panel_plot(summary, cfg.OUT_DIR, dom, "layer", summary.alpha == A0, "слой", f"pT_vs_layer_{dom}_A{A0}.png")
    log("графики сохранены")
