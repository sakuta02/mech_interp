"""Анализ результатов опыта: фильтр примеров, сводные таблицы, доверительные интервалы и парные тесты по raw.csv.
Работает без torch и GPU, поэтому его можно запускать на ноутбуке."""
import os
import pandas as pd
from scipy.stats import binomtest

KEY = ["tool_id", "trans", "tpl", "ex"]
PAIRS = [("steer_tool", "steer_foreign"), ("steer_tool", "shuffled"), ("steer_foreign", "shuffled"), ("steer_tool", "steer_neg")]


def clean(df):
    """Оставляет примеры, где модель верна без вмешательства (свой слот и все предыдущие аргументы); считает сдвиг и долю прочих ответов."""
    dd = df[df.base_ok & df.ctx_ok & (df.kind != "none")].copy()
    dd["other"] = ~(dd.top1_is_T | dd.top1_is_S | dd.top1_is_R)
    for col in ["top1_is_T", "top1_is_S", "top1_is_R", "other"]:
        dd[col] = dd[col].astype(float)
    dd["shift"] = dd.delta - dd.base_delta
    return dd


def summarize(cfg, df, dd, log):
    """Сводка по (инструмент, переход, вид условия, сила, слой) и по всем инструментам сразу (ALL); пишет summary.csv и base_rates.csv."""
    out = cfg.OUT_DIR
    agg = dict(pT=("top1_is_T", "mean"), pS=("top1_is_S", "mean"), pR=("top1_is_R", "mean"),
               other=("other", "mean"), shift=("shift", "mean"), n=("shift", "size"))
    by = ["trans", "kind", "alpha", "layer"]
    per_tool = dd.groupby(["tool_id"] + by, dropna=False).agg(**agg).reset_index()
    pooled = dd.groupby(by, dropna=False).agg(**agg).reset_index().assign(tool_id="ALL")
    summary = pd.concat([per_tool, pooled], ignore_index=True)
    summary.to_csv(f"{out}/summary.csv", index=False)

    base_rates = (df[df.layer == -1].groupby(["tool_id", "trans"]).agg(base_ok=("base_ok", "mean"), ctx_ok=("ctx_ok", "mean"),
                                                                       n=("base_ok", "size")).reset_index())
    base_rates.to_csv(f"{out}/base_rates.csv", index=False)
    print(base_rates.round(3).to_string(index=False))
    log("сводка сохранена")
    return summary


def wilson(g):
    """Добавляет доверительный интервал Уилсона для доли mean = sum / size."""
    ci = [binomtest(int(r["sum"]), int(r["size"])).proportion_ci(method="wilson") for _, r in g.iterrows()]
    g["lo"], g["hi"] = [x.low for x in ci], [x.high for x in ci]
    return g


def cell_stats(cfg, dd):
    """Доля целевого значения при слое L0 и силе A0 по ячейкам и разрыв gap = steer_tool − steer_foreign."""
    q = dd[(dd.layer == cfg.L0) & (dd.alpha == cfg.A0)]
    agg = lambda keys: q.groupby(keys).top1_is_T.agg(["mean", "sum", "size"]).reset_index()
    g = pd.concat([agg(["tool_id", "trans", "kind"]), agg(["trans", "kind"]).assign(tool_id="ALL")], ignore_index=True)
    g = wilson(g)
    g.to_csv(f"{cfg.OUT_DIR}/cell_stats_L{cfg.L0}_A{cfg.A0}.csv", index=False)
    pv = g.pivot_table(index=["tool_id", "trans"], columns="kind", values="mean")
    if {"steer_tool", "steer_foreign"} <= set(pv.columns):
        pv["gap"] = pv.steer_tool - pv.steer_foreign
    pv.round(3).to_csv(f"{cfg.OUT_DIR}/gap_L{cfg.L0}_A{cfg.A0}.csv")
    print(f"доля целевого значения, слой {cfg.L0}, alpha {cfg.A0}")
    print(pv.round(3).to_string())
    print("\nn (steer_tool):")
    print(g[g.kind == "steer_tool"].set_index(["tool_id", "trans"])["size"].to_string(), "\n")
    return g


def mcnemar(dd, layer, alpha, ca, cb, **flt):
    """Точный парный тест McNemar для видов условий ca и cb на одних и тех же примерах; None, если сравнивать нечего."""
    q = dd[(dd.layer == layer) & (dd.alpha == alpha) & dd.kind.isin([ca, cb])]
    for k, v in flt.items():
        q = q[q[k] == v]
    if q.empty or q.kind.nunique() < 2:
        return None
    p = q.pivot_table(index=KEY, columns="kind", values="top1_is_T", aggfunc="max").dropna() > 0
    b_, c_ = int((p[ca] & ~p[cb]).sum()), int((~p[ca] & p[cb]).sum())
    return dict(**flt, layer=layer, alpha=alpha, pair=f"{ca} vs {cb}", n=len(p),
                a=round(p[ca].mean(), 3), b=round(p[cb].mean(), 3), only_a=b_, only_b=c_,
                p=binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0)


def paired_tests(cfg, dd):
    """Парные сравнения видов условий при L0 и A0: по всем инструментам и по каждому отдельно; пишет mcnemar_*.csv."""
    L0, A0 = cfg.L0, cfg.A0
    trans = sorted(dd.trans.unique())
    res = [mcnemar(dd, L0, A0, ca, cb, trans=tr) for tr in trans for ca, cb in PAIRS]                      # по всем инструментам
    res += [mcnemar(dd, L0, A0, ca, cb, trans=tr, tool_id=ti) for ti in sorted(dd.tool_id.unique())          # по каждому инструменту
            for tr in trans for ca, cb in PAIRS[:2]]
    mc = pd.DataFrame([r for r in res if r]).round(4)
    mc.to_csv(f"{cfg.OUT_DIR}/mcnemar_L{L0}_A{A0}.csv", index=False)
    print("парные сравнения (без поправки на множественные сравнения):")
    print(mc.to_string(index=False))
    return mc


def where_goes(cfg, dd):
    """Куда уходит выдача при вмешательстве: целевое значение T, своё S, значения остальных аргументов R или прочее."""
    q = dd[(dd.layer == cfg.L0) & (dd.alpha == cfg.A0) & dd.kind.isin(["steer_tool", "steer_foreign"])]
    grp = q.groupby(["tool_id", "trans", "kind"])
    t_ = grp[["top1_is_T", "top1_is_S", "top1_is_R", "other"]].mean().round(3)
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
    return dd, summarize(cfg, df, dd, log)


def stats(cfg, dd, log):
    """Таблицы по ячейкам, парные тесты и распределение выдачи при L0 и A0."""
    cell_stats(cfg, dd)
    paired_tests(cfg, dd)
    log("статистика сохранена")
    where_goes(cfg, dd)
    print(sorted(os.listdir(cfg.OUT_DIR)))
