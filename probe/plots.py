"""Графики: точность проб по слоям для каждого вопроса, наборы в разных цветах, контроль и бейзлайны."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_plots(cfg, res, log):
    """Сетка графиков: логрег по наборам, перемешанные метки (серая пунктирная), бейзлайн по позиции и по названиям."""
    qs = list(res.question.unique())
    nc = 3
    fig, axs = plt.subplots((len(qs) + nc - 1) // nc, nc, figsize=(6 * nc, 3.6 * ((len(qs) + nc - 1) // nc)), squeeze=False)
    for ax, q in zip(axs.flat, qs):
        d = res[res.question == q]
        lr = d[(d.probe == "logreg") & (d.layer >= 0)]
        for s, x in lr.groupby("set"):
            x = x.sort_values("layer")
            ax.plot(x.layer, x.bacc, marker=".", lw=2 if s == "iid_test" else 1, label=s)
            if s == "iid_test":
                ax.fill_between(x.layer, x.lo, x.hi, alpha=0.2)
        sh = d[d.probe.str.startswith("shuf") & (d.set == "iid_test")].groupby("layer").bacc.mean()
        ax.plot(sh.index, sh.values, "--", color="gray", label="shuffled")
        for kind, c in [("pos", "k"), ("lex", "r")]:
            b = d[(d.probe == kind) & (d.set == "iid_test") & (d.layer == -1)]
            if len(b):
                ax.axhline(b.bacc.iloc[0], ls=":", color=c, label=f"{kind} (iid)")
        ax.set_title(q, fontsize=9); ax.set_xlabel("слой"); ax.set_ylabel("bacc"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    axs.flat[0].legend(fontsize=6)
    for ax in list(axs.flat)[len(qs):]:
        ax.axis("off")
    fig.suptitle("пробы: сбалансированная точность по слоям")
    fig.tight_layout()
    fig.savefig(f"{cfg.OUT_DIR}/probes.png", dpi=110)
    plt.close(fig)
    log("графики сохранены")
