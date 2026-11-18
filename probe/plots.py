"""Графики: по одному файлу на вопрос опыта, на каждом немного линий. Цвет закреплён за набором, чтобы читать одинаково
на всех рисунках. Серая пунктирная — перемешанные метки; горизонтальные точечные — бейзлайны по позиции и по названиям полей."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"iid_test": "#1f77b4", "ood_all": "#ff7f0e", "ood_name": "#2ca02c", "ood_desc": "#9467bd", "ood_tpl": "#8c564b",
          "ood_city": "#e377c2", "err_slot": "#d62728", "err_model": "#d62728", "err_gold": "#7f7f7f",
          "swap_field": "#17becf", "swap_city": "#bcbd22", "xslot": "#d62728", "same_slot": "#1f77b4",
          "conflict_key": "#2ca02c", "conflict_desc": "#9467bd", "dist1": "#1f77b4", "dist2": "#ff7f0e", "dist3": "#d62728"}
LABELS = {"err_model": "ошибочные слоты: согласие с выбором модели", "err_slot": "ошибочные слоты (золотая роль)",
          "swap_field": "swap: метка по полю", "swap_city": "swap: метка по городу", "xslot": "другой вид слота",
          "same_slot": "тот же вид слота", "conflict_key": "conflict: по названию поля", "conflict_desc": "conflict: по описанию",
          "dist1": "поле 1 слот назад", "dist2": "2 слота назад", "dist3": "3 слота назад"}


def panel(ax, res, q, sets, baselines=True, shuffled=True, ylabel="bacc"):
    """Одна панель: логрег по слоям для наборов sets (интервал у первого набора), контроль и бейзлайны."""
    d = res[res.question == q]
    lr = d[(d.probe == "logreg") & (d.layer >= 0)]
    for i, s in enumerate(sets):
        x = lr[lr.set == s].sort_values("layer")
        if x.empty:
            continue
        ax.plot(x.layer, x.bacc, marker=".", lw=2 if i == 0 else 1.2, color=COLORS.get(s), label=LABELS.get(s, s))
        if i == 0:
            ax.fill_between(x.layer, x.lo, x.hi, alpha=0.15, color=COLORS.get(s))
    if shuffled:
        sh = d[d.probe.str.startswith("shuf") & (d.set == sets[0])].groupby("layer").bacc.mean()
        ax.plot(sh.index, sh.values, "--", color="gray", label="перемешанные метки")
    if baselines:
        for kind, c in [("pos", "k"), ("lex", "r")]:
            b = d[(d.probe == kind) & (d.set == sets[0])]
            if len(b):
                ax.axhline(b.bacc.iloc[0], ls=":", color=c, label="бейзлайн: позиция" if kind == "pos" else "бейзлайн: названия")
    ax.set_title(q, fontsize=10); ax.set_xlabel("слой"); ax.set_ylabel(ylabel); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def figure(cfg, name, title, panels, ncols=2, **kw):
    """Файл из нескольких панелей: panels = [(вопрос, наборы)]; kw передаются в panel."""
    nr = (len(panels) + ncols - 1) // ncols
    fig, axs = plt.subplots(nr, ncols, figsize=(6.5 * ncols, 4 * nr), squeeze=False)
    for ax, (q, sets) in zip(axs.flat, panels):
        panel(ax, cfg["res"], q, sets, **kw)
    for ax in list(axs.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{cfg['out']}/{name}.png", dpi=110)
    plt.close(fig)


def readout_figures(cfg, res, cosdf, key_rate, log):
    """Все рисунки проб; key_rate — доля конфликтных слотов, где модель пишет город по названию поля."""
    c = dict(res=res, out=cfg.OUT_DIR)
    std = ["iid_test", "ood_name", "ood_desc", "ood_tpl", "ood_all"]
    figure(c, "01_controls", "Контроли: Q0_marker — роль по маркеру перед городом (баг, ожидаем 1); "
           "Q0_leak — первый город postfix (утечка, ожидаем 0.5)", [("Q0_marker", ["iid_test", "ood_tpl", "ood_all"]),
                                                                ("Q0_leak", ["iid_test", "ood_tpl", "ood_all"])])
    figure(c, "02_role", "Роль поля (src / dst / прочее) и номер слота", [("Q1_role", std), ("Q2_ordinal", ["iid_test", "ood_name", "ood_all"])])
    figure(c, "03_pointer", "Указатель: какой город будет скопирован (по роли и по порядку упоминания)",
           [("Q3_ptr_role", ["iid_test", "ood_name", "ood_all"]), ("Q3_ptr_pos", ["iid_test", "ood_tpl", "ood_all"])])
    figure(c, "04_pointer_errors", "Слоты, где модель выбрала город другой роли: идёт ли проба за выбором модели (1) или за золотом (0)",
           [("Q3_ptr_role", ["err_model"]), ("Q3_ptr_pos", ["err_model"])], baselines=False, shuffled=False,
           ylabel="согласие с выбором модели")
    figure(c, "05_used", "Учёт записанного на слоте 2: использован ли src / dst",
           [("Q4_used_src", ["iid_test", "ood_name", "ood_all"]), ("Q4_used_dst", ["iid_test", "ood_name", "ood_all"])])
    figure(c, "06_used_swap", "Swap: первое поле заполнено городом другой роли. Проба идёт за полем или за городом?",
           [("Q4_used_src", ["swap_field", "swap_city"]), ("Q4_used_dst", ["swap_field", "swap_city"])], baselines=False, shuffled=False)
    figure(c, "07_used_xslot", "Перенос «использован» между видами слота 2 (o2c: обучение на посторонних, тест на городском; c2o — наоборот)",
           [(f"Q4_used_{r}_{d}", ["xslot", "same_slot"]) for r in ("src", "dst") for d in ("o2c", "c2o")])
    figure(c, "08_used_distance", "Учёт записанного на слотах 2–4 в зависимости от того, сколько полей назад записано поле роли",
           [("Q4d_used_src", ["dist1", "dist2", "dist3"]), ("Q4d_used_dst", ["dist1", "dist2", "dist3"])], baselines=False)
    _conflict(c, key_rate)
    _saturation(c)
    _dirs(cfg, cosdf)
    log("графики проб сохранены")


def _conflict(c, key_rate):
    """Конфликт название ↔ описание: за чем идёт проба и за чем идёт сама модель (горизонтальная линия)."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for ax, q in zip(axs, ["Q1_role", "Q3_ptr_role"]):
        panel(ax, c["res"], q, ["conflict_key", "conflict_desc"], baselines=False, shuffled=False, ylabel="доля согласия")
        if not np.isnan(key_rate):
            ax.axhline(key_rate, color="k", lw=1.5, label="модель пишет город по названию")
            ax.legend(fontsize=7)
    fig.suptitle("Конфликт: название поля одной роли, описание другой")
    fig.tight_layout(); fig.savefig(f"{c['out']}/09_conflict.png", dpi=110); plt.close(fig)


def _saturation(c):
    """Тепловая карта: первый слой, где bacc ≥ 0.95, по вопросам и наборам."""
    res = c["res"]
    sets = ["iid_test", "ood_city", "ood_name", "ood_desc", "ood_tpl", "ood_all"]
    qs = [q for q in res.question.unique() if "_o2c" not in q and "_c2o" not in q]
    lr = res[(res.probe == "logreg") & (res.layer >= 0)]
    M = np.full((len(qs), len(sets)), np.nan)
    for i, q in enumerate(qs):
        for j, s in enumerate(sets):
            x = lr[(lr.question == q) & (lr.set == s) & (lr.bacc >= 0.95)]
            if len(x):
                M[i, j] = x.layer.min()
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(qs) + 1.5))
    im = ax.imshow(np.nan_to_num(M, nan=-1), cmap="viridis", vmin=-1, aspect="auto")
    for i in range(len(qs)):
        for j in range(len(sets)):
            ax.text(j, i, "—" if np.isnan(M[i, j]) else f"{M[i, j]:.0f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_xticks(range(len(sets)), sets, rotation=30); ax.set_yticks(range(len(qs)), qs)
    ax.set_title("Первый слой, где bacc ≥ 0.95 («—»: не достигается)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(f"{c['out']}/10_saturation.png", dpi=110); plt.close(fig)


def _dirs(cfg, cosdf):
    """Косинус между направлением логрега и разностью средних по слоям (бинарные вопросы)."""
    if cosdf is None or cosdf.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for q, x in cosdf.groupby("question"):
        if "_o2c" in q or "_c2o" in q:
            continue
        ax.plot(x.layer, x.cos, marker=".", label=q)
    ax.set_xlabel("слой"); ax.set_ylabel("cos"); ax.set_ylim(-0.05, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title("Согласие направлений логрега и разности средних")
    fig.tight_layout(); fig.savefig(f"{cfg.OUT_DIR}/11_dirs.png", dpi=110); plt.close(fig)


PATCH_TITLES = {"marker": "Контрфакт marker: from ↔ to в сообщении, города на месте",
                "city": "Контрфакт city: города меняются местами, маркеры на месте",
                "schema": "Контрфакт schema: у городских полей меняются местами названия и описания"}
GROUP_COLORS = {"slot": "#1f77b4", "cities": "#ff7f0e", "markers": "#2ca02c", "key": "#d62728"}
HOW_STYLE = {"full": "-", "dir": "--", "rand": ":"}
HOW_LABEL = {"full": "целиком", "dir": "вдоль направления пробы", "rand": "вдоль случайного"}


def patch_figures(cfg, pr, log):
    """По файлу на контрфакт: слева нормированная разность логитов, справа доля переворота ответа, по слоям."""
    for i, (kind, d) in enumerate(pr.groupby("kind", sort=False)):
        fig, axs = plt.subplots(1, 2, figsize=(13, 4))
        for (g, h), x in d.groupby(["group", "how"], sort=False):
            m = x.groupby("layer")[["nld", "flip"]].mean()
            for ax, col in zip(axs, ["nld", "flip"]):
                ax.plot(m.index, m[col], HOW_STYLE[h], marker=".", color=GROUP_COLORS[g], label=f"{g}: {HOW_LABEL[h]}")
        for ax, yl in zip(axs, ["нормированная разность логитов", "доля переворота ответа"]):
            ax.set_xlabel("слой"); ax.set_ylabel(yl); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        axs[1].set_ylim(-0.02, 1.02)
        fig.suptitle(f"{PATCH_TITLES[kind]} (пар: {d.ex.nunique()})")
        fig.tight_layout(); fig.savefig(f"{cfg.OUT_DIR}/{12 + i}_patch_{kind}.png", dpi=110); plt.close(fig)
    log("графики патчинга сохранены")
