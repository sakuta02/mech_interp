"""Анализ результатов: фильтр примеров, сводные таблицы, доверительные интервалы и парные тесты по таблице raw.csv.
Работает без torch и GPU, поэтому его можно запускать на ноутбуке.
"""
import os
import pandas as pd
from scipy.stats import binomtest

from data import LAYOUTS

KEY = ["tool_id", "tpl", "slot", "O", "D", "X"]
SL = ("first", "second")


def clean(df):
    """Оставляет примеры, где модель верна без вмешательства (во втором слоте ещё и первое значение записано верно); считает сдвиг и долю прочих ответов."""
    ok_first = (df.slot == "first") | (df.first_ok == True)
    dd = df[df.base_ok & ok_first & (df.cond != "none")].copy()
    dd["cond"] = dd.cond.str.replace(r"random\d+", "random", regex=True)
    dd["other"] = ~(dd.top1_is_T | dd.top1_is_S | dd.top1_is_X)
    for col in ["top1_is_T", "top1_is_S", "top1_is_X", "other"]:
        dd[col] = dd[col].astype(float)
    dd["shift"] = dd.delta - dd.base_delta
    return dd


def summarize(cfg, df, dd, log):
    """Сводка по ячейкам (тип задач, условие, сила, слой, слот, порядок, вариант вызова) и доля верных базовых ответов; пишет summary.csv и base_rates.csv."""
    out = cfg.OUT_DIR
    summary = (dd.groupby(["domain", "cond", "alpha", "layer", "slot", "order", "layout"], dropna=False)
                 .agg(pT=("top1_is_T", "mean"), pS=("top1_is_S", "mean"), pX=("top1_is_X", "mean"),
                      other=("other", "mean"), shift=("shift", "mean"), n=("shift", "size")).reset_index())
    summary.to_csv(f"{out}/summary.csv", index=False)

    base_rates = df[df.layer == -1].groupby(["domain", "layout", "order", "slot"]).base_ok.agg(["mean", "size"]).reset_index()
    base_rates.to_csv(f"{out}/base_rates.csv", index=False)
    print(base_rates.round(3).to_string(index=False))
    log("сводка сохранена")
    return summary


def cell_stats(cfg, dd):
    """Доля целевого значения при слое L0 и силе A0 с доверительным интервалом Уилсона по каждой ячейке."""
    q = dd[(dd.layer == cfg.L0) & (dd.alpha == cfg.A0)]
    g = q.groupby(["domain", "layout", "order", "slot", "cond"]).top1_is_T.agg(["mean", "sum", "size"]).reset_index()
    ci = [binomtest(int(r["sum"]), int(r["size"])).proportion_ci(method="wilson") for _, r in g.iterrows()]
    g["lo"], g["hi"] = [x.low for x in ci], [x.high for x in ci]
    g.to_csv(f"{cfg.OUT_DIR}/cell_stats_L{cfg.L0}_A{cfg.A0}.csv", index=False)
    idx = ["domain", "layout", "order", "slot"]
    print(f"доля целевого значения, слой {cfg.L0}, alpha {cfg.A0}")
    print(g.pivot_table(index=idx, columns="cond", values="mean").round(3).to_string())
    print("\nn (steer):")
    print(g[g.cond == "steer"].set_index(idx)["size"].to_string(), "\n")
    return g


def mcnemar(dd, layer, alpha, ca, cb, **flt):
    """Точный парный тест McNemar для условий ca и cb на одних и тех же примерах; None, если сравнивать нечего."""
    q = dd[(dd.layer == layer) & (dd.alpha == alpha) & dd.cond.isin([ca, cb])]
    for k, v in flt.items():
        q = q[q[k] == v]
    if q.empty or q.cond.nunique() < 2:
        return None
    p = q.pivot_table(index=KEY, columns="cond", values="top1_is_T", aggfunc="max").dropna() > 0
    b_, c_ = int((p[ca] & ~p[cb]).sum()), int((~p[ca] & p[cb]).sum())
    return dict(**flt, layer=layer, alpha=alpha, pair=f"{ca} vs {cb}", n=len(p),
                a=round(p[ca].mean(), 3), b=round(p[cb].mean(), 3), only_a=b_, only_b=c_,
                p=binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0)


def paired_tests(cfg, dd):
    """Набор парных сравнений условий при L0 и A0; пишет mcnemar_*.csv."""
    L0, A0 = cfg.L0, cfg.A0
    res = [mcnemar(dd, L0, A0, ca, cb, domain="flight", layout=lay, order=o, slot=sl)          # дата в вызове и порядок слотов
           for lay in LAYOUTS for o in ("od", "do") for sl in SL
           for ca, cb in [("steer_same", "steer_layout"), ("steer_same", "steer_opp")]]
    res += [mcnemar(dd, L0, A0, "steer", "shuffled", domain="flight", slot=sl) for sl in SL]
    res += [mcnemar(dd, L0, A0, ca, cb, domain=dom, slot=sl)                                   # перенос в другие типы задач
            for dom in ("message", "period") for sl in SL
            for ca, cb in [("steer", "steer_dom"), ("steer", "shuffled"), ("steer_same", "steer_opp")]]
    mc = pd.DataFrame([r for r in res if r]).round(4)
    mc.to_csv(f"{cfg.OUT_DIR}/mcnemar_L{L0}_A{A0}.csv", index=False)
    print("парные сравнения (без поправки на множественные сравнения):")
    print(mc.to_string(index=False))
    return mc


def where_goes(cfg, dd):
    """Куда уходит выдача при вмешательстве: целевое значение, своё, третье или прочее."""
    q = dd[(dd.layer == cfg.L0) & (dd.alpha == cfg.A0) &
           dd.cond.isin(["steer", "steer_same", "steer_opp", "steer_layout", "steer_dom"])]
    grp = q.groupby(["domain", "slot", "order", "cond"])
    t_ = grp[["top1_is_T", "top1_is_S", "top1_is_X", "other"]].mean().round(3)
    t_["n"] = grp.size()
    t_.to_csv(f"{cfg.OUT_DIR}/where_goes_L{cfg.L0}_A{cfg.A0}.csv")
    print(t_.to_string())
    return t_


def analyze(cfg, df, log):
    """Фильтрует данные и строит сводку; None, если после фильтра примеров не осталось."""
    dd = clean(df)
    if dd.empty:
        log("ПРЕДУПРЕЖДЕНИЕ: после фильтра (модель верна без вмешательства) не осталось примеров, анализ пропущен")
        return None
    summary = summarize(cfg, df, dd, log)
    return dd, summary


def stats(cfg, dd, log):
    """Таблицы по ячейкам, парные тесты и распределение выдачи при L0 и A0."""
    cell_stats(cfg, dd)
    paired_tests(cfg, dd)
    log("статистика сохранена")
    where_goes(cfg, dd)
    print(sorted(os.listdir(cfg.OUT_DIR)))
