"""Сводка результатов проб: лучший слой каждого вопроса и сравнение наборов, контролей и бейзлайнов на этом слое."""
import numpy as np, pandas as pd


def _bacc(d, probe, s, layer):
    """Сбалансированная точность выбранной пробы на наборе s и слое layer (среднее, если строк несколько)."""
    x = d[(d.layer == layer) & (d.set == s) & d.probe.str.startswith(probe)]
    return float(x.bacc.mean()) if len(x) else np.nan


def summarize(res, log):
    """Таблица: на лучшем (по iid_test) слое логрега — точность по всем наборам, dim, shuf, pos, lex и селективность."""
    rows = []
    for q, d in res.groupby("question", sort=False):
        lr = d[(d.probe == "logreg") & (d.set == "iid_test") & (d.layer >= 0)].dropna(subset=["bacc"])
        if lr.empty:
            continue
        l = int(lr.loc[lr.bacc.idxmax(), "layer"])
        r = dict(question=q, best_layer=l)
        for s in sorted(d.set.unique()):
            r[s] = _bacc(d, "logreg", s, l)
        r.update(dim_iid=_bacc(d, "dim", "iid_test", l), shuf_iid=_bacc(d, "shuf", "iid_test", l),
                 pos_iid=_bacc(d, "pos", "iid_test", -1), lex_iid=_bacc(d, "lex", "iid_test", -1),
                 lex_ood_name=_bacc(d, "lex", "ood_name", -1))
        r["selectivity"] = r["iid_test"] - r["shuf_iid"]
        rows.append(r)
    out = pd.DataFrame(rows)
    for _, r in out.iterrows():
        log("итог:", r.question, "слой", int(r.best_layer), " ".join(f"{k}={v:.2f}" for k, v in r.items()
            if k not in ("question", "best_layer") and not np.isnan(v)))
    return out
