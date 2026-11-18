"""Сводки: по пробам — первый слой, где набор насыщается (bacc ≥ 0.95), и точность на опорном слое; по патчингу — максимум
эффекта по слоям; по поведению модели — ошибки и выбор в конфликтных примерах."""
import numpy as np, pandas as pd


def first_layer(x, thr=0.95):
    """Первый слой, где bacc ≥ thr; NaN, если не достигается."""
    x = x[x.bacc >= thr]
    return float(x.layer.min()) if len(x) else np.nan


def summarize(res, log, ref=20):
    """Таблица по вопросам: L95_<набор> — первый слой насыщения; <набор>@L — bacc на опорном слое ref (или ближайшем);
    контроли shuf, pos, lex на iid_test."""
    rows = []
    for q, d in res.groupby("question", sort=False):
        lr = d[(d.probe == "logreg") & (d.layer >= 0)]
        if lr.empty:
            continue
        L = int(min(lr.layer.unique(), key=lambda l: abs(l - ref)))
        r = dict(question=q, ref_layer=L)
        for s, x in lr.groupby("set"):
            r[f"L95_{s}"] = first_layer(x)
            r[f"{s}@L"] = float(x[x.layer == L].bacc.mean())
        main = "iid_test" if "iid_test" in lr.set.values else lr.set.iloc[0]
        sh = d[d.probe.str.startswith("shuf") & (d.set == main) & (d.layer == L)].bacc
        r["shuf@L"] = float(sh.mean()) if len(sh) else np.nan
        for kind in ("pos", "lex"):
            b = d[(d.probe == kind) & (d.set == main)].bacc
            r[kind] = float(b.iloc[0]) if len(b) else np.nan
        rows.append(r)
        log(f"итог {q}: первый слой ≥0.95: " + " ".join(f"{k[4:]}={v:.0f}" for k, v in r.items() if k.startswith("L95_") and not np.isnan(v))
            + f" | слой {L}: " + " ".join(f"{k[:-2]}={v:.2f}" for k, v in r.items() if k.endswith("@L") and not np.isnan(v))
            + f" | pos={r['pos']:.2f} lex={r['lex']:.2f}")
    return pd.DataFrame(rows)


def behavior(df, log):
    """Поведение модели: доля верных примеров по группам, число ошибочных слотов, выбор в конфликтных примерах.
    Возвращает долю городских слотов conflict, где модель пишет город по названию поля (а не по описанию)."""
    acc = df.groupby(["group", "ex"]).correct.first().groupby("group").mean()
    log("доля примеров, где модель верна в городских слотах:", {g: round(v, 3) for g, v in acc.items() if g not in ("swap", "conflict")})
    log("ошибочных городских слотов (модель выбрала город другой роли):", int(df.err_slot.sum()))
    c = df[(df.group == "conflict") & df.cur_city & (df.site < 4) & (df.ptr_role >= 0)]
    key = float((c.ptr_role == c.y_role).mean()) if len(c) else np.nan
    log(f"конфликт название ↔ описание: модель пишет город по названию поля в {key:.2f} слотов, по описанию в {1 - key:.2f}")
    return key


def summarize_patch(pr, log):
    """По каждому (контрфакт, группа, как): максимум среднего nld и средней доли переворота по слоям и их слои."""
    rows = []
    for (k, g, h), d in pr.groupby(["kind", "group", "how"], sort=False):
        m = d.groupby("layer")[["nld", "flip"]].mean()
        rows.append(dict(kind=k, group=g, how=h, n=d.ex.nunique(), nld_max=m.nld.max(), nld_layer=int(m.nld.idxmax()),
                         flip_max=m.flip.max(), flip_layer=int(m.flip.idxmax())))
        log(f"патч {k:6s} {g:8s} {h:4s}: nld max {m.nld.max():.2f} (слой {int(m.nld.idxmax())}), "
            f"переворот max {m.flip.max():.2f} (слой {int(m.flip.idxmax())}), пар {d.ex.nunique()}")
    return pd.DataFrame(rows)
