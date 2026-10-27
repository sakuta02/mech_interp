"""Проверки режима --test: на каждом этапе опыта проверяют формы, нормы, работу хуков и итоговые файлы и пишут в лог OK, FAIL или WARN."""
import os, random
from dataclasses import replace
import pandas as pd, torch

import analysis, plots
from data import TOOLS, TEMPLATES, LAYOUTS
from evaluate import eval_example
from model import RESID, steer
from prompts import arg_seq, tf_ids, prefix, probe_format
from vectors import rand_unit


class Checker:
    """Собирает результаты проверок: провалы (FAIL) и предупреждения (WARN)."""
    def __init__(self, log):
        """Запоминает лог и заводит списки провалов и предупреждений."""
        self.log, self.failed, self.warned = log, [], []

    def __call__(self, name, cond, detail=""):
        """Записывает результат проверки; ложное условие считается провалом."""
        self.log(f"[test] {'OK  ' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
        if not cond:
            self.failed.append(name)

    def warn(self, name, cond, detail=""):
        """То же, но ложное условие даёт только предупреждение."""
        self.log(f"[test] {'OK  ' if cond else 'WARN'} {name}" + (f"  ({detail})" if detail else ""))
        if not cond:
            self.warned.append(name)

    def report(self):
        """Печатает итог; True, если провалов нет."""
        self.log(f"[test] итог: провалено {len(self.failed)}, предупреждений {len(self.warned)}"
                 + (f"; провалены: {self.failed}" if self.failed else ""))
        return not self.failed


def check_data(ck, lm, pool):
    """Проверяет набор инструментов, шаблоны и пулы значений."""
    ck("44 инструмента (36 рейсов + 4 письма + 4 периода)", len(TOOLS) == 44, f"{len(TOOLS)}")
    ck("id инструментов уникальны", len({t["id"] for t in TOOLS}) == len(TOOLS))
    ck("в каждом шаблоне есть {O} и {D}", all("{O}" in x["text"] and "{D}" in x["text"] for x in TEMPLATES))
    for dom in ("flight", "message", "period"):
        ck(f"{dom}: есть калибровочные и тестовые шаблоны",
           {x["split"] for x in TEMPLATES if x["domain"] == dom} == {"calib", "test"})
        ck(f"{dom}: пулы калибровки и теста не пересекаются", not set(pool[dom]["calib"]) & set(pool[dom]["test"]))
        ck(f"{dom}: все значения пула однотокенные", all(len(lm.enc(v)) == 1 for k in ("calib", "test") for v in pool[dom][k]))
        ck(f"{dom}: в пуле хватает значений для random.sample(3)", min(len(pool[dom]["calib"]), len(pool[dom]["test"])) >= 3)


def check_prompts(ck, lm, pool):
    """Проверяет сборку промптов: позиции слотов и согласованность prefix с teacher forcing."""
    bad_pos, bad_pref, bad_role, n = 0, 0, 0, 0
    for t in TOOLS:
        for tpl in (x for x in TEMPLATES if x["domain"] == t["domain"]):
            O, D, X = pool[t["domain"]]["calib"][:3]
            ids, pos = tf_ids(lm, t, tpl, O, D, X)
            seq = [int(v) for v in ids[0]]
            n += 1
            bad_pos += any(seq[pos[r] + 1] != lm.enc({"o": O, "d": D}[r])[0] for r in pos)
            roles = [r for r, _ in arg_seq(t) if r != "x"]
            p1, role1, tok1 = prefix(lm, t, tpl, O, D, X, "first")
            bad_pref += p1 != seq[:pos[role1] + 1] or tok1 is not None
            bad_role += role1 != roles[0]
    ck(f"tf_ids: токен после pos[роль] совпадает со значением ({n} комбинаций)", bad_pos == 0, f"ошибок {bad_pos}")
    ck("prefix(first) совпадает с префиксом teacher forcing", bad_pref == 0, f"ошибок {bad_pref}")
    ck("prefix(first): роль равна роли первого города вызова", bad_role == 0, f"ошибок {bad_role}")
    t = next(t for t in TOOLS if t["layout"] == "date_mid")
    ck("date_mid: дата между двумя городами", [r for r, _ in arg_seq(t)][1] == "x")
    ck("date_first: дата первой", [r for r, _ in arg_seq(next(t for t in TOOLS if t["layout"] == "date_first"))][0] == "x")


def check_model(ck, lm, cfg, pool):
    """Проверяет чтение активаций хуком и правку активаций хуками на загруженной модели."""
    t, tpl = TOOLS[0], TEMPLATES[0]
    O, D, X = pool["flight"]["calib"][:3]
    ids, pos = tf_ids(lm, t, tpl, O, D, X)
    T = ids.shape[1]
    logits, st = lm.run_states(ids, pos)
    ck("run_states: логиты [T, V]", logits.shape[0] == T and logits.ndim == 2, f"{tuple(logits.shape)}")
    ck("run_states: состояния [слои, d] для o и d", all(v.shape[0] == lm.n_layers and v.ndim == 2 for v in st.values())
       and set(st) == {"o", "d"}, f"{ {r: tuple(v.shape) for r, v in st.items()} }")
    ck("run_states: нет NaN/inf", all(torch.isfinite(v).all().item() for v in st.values()))
    _, cache = lm.model.run_with_cache(ids, names_filter=[RESID(l) for l in range(lm.n_layers)])
    ck("читающий хук даёт те же активации, что run_with_cache (бит в бит)",
       all(torch.equal(st[r], torch.stack([cache[RESID(l)][0, p] for l in range(lm.n_layers)]).float()) for r, p in pos.items()))
    ck("run_states(logits=False) не считает логиты", lm.run_states(ids, {"o": 0}, logits=False)[0] is None)

    x = lm.tensor(prefix(lm, t, tpl, O, D, X, "first")[0])
    base = lm.fwd_last(x)
    l = cfg.L0
    d = lambda a, b: (a - b).abs().max().item()
    ck("fwd_last: форма [1, V]", base.ndim == 2 and base.shape[0] == 1, f"{tuple(base.shape)}")
    _, stx = lm.run_states(x, {"p": x.shape[1] - 1})
    dl = d(lm.model(x)[0, -1][None].float(), base)
    ck("model(x) и fwd_last согласованы по последней позиции", dl < 0.5, f"макс. разница логитов {dl:.3f}")

    zero = torch.zeros(1, stx["p"].shape[-1], device=lm.device)
    ck("хук steer(adds=0) не меняет логиты", d(lm.fwd_last(x, [(RESID(l), steer(adds=zero))]), base) < 1e-3)
    ck("хук steer(adds=вектор) меняет логиты", d(lm.fwd_last(x, [(RESID(l), steer(adds=stx["p"][l][None] * 0.5))]), base) > 0)
    ck("хуки снимаются после прогона", d(lm.fwd_last(x), base) < 1e-3)
    ck("подмена собственной активацией (repl) не меняет логиты", d(lm.fwd_last(x, [(RESID(l), steer(repl=stx["p"][l]))]), base) < 1e-3)
    lg3 = lm.fwd_last(x.repeat(3, 1))
    ck("батч из одинаковых строк даёт те же логиты, что одна строка", max(d(lg3[i:i + 1], base) for i in range(3)) < 1.0,
       f"макс. разница {max(d(lg3[i:i + 1], base) for i in range(3)):.3f}")
    ck("generate отрабатывает и возвращает строку", isinstance(probe_format(lm, lambda *a: None), str))


def check_vectors(ck, cfg, lm, H, V, vecs):
    """Проверяет формы, нормы и ортогональность векторов и контролей.
    Слой 0 исключён: там resid_pre — эмбеддинг одного токена перед обоими слотами, s = 0 и нормировка даёт NaN; стиринг идёт только по LAYER_RANGE.
    """
    L = lm.n_layers
    ck("калибровка: [kept, слои, d] для каждого инструмента", all(a.ndim == 3 and a.shape[1] == L and a.shape == b.shape
                                                                  and a.shape[0] >= 1 for a, b in H.values()))
    ck("s: [слои, d] без NaN/inf", V.s.shape[0] == L and torch.isfinite(V.s).all().item())
    ck("D_cfg: 2 порядка × 3 варианта вызова", len(V.D_cfg) == 6 and set(V.D_cfg) == {(o, lay) for o in ("od", "do") for lay in LAYOUTS})
    lr = list(cfg.LAYER_RANGE)
    ck("d_shuf без NaN/inf на рабочих слоях", torch.isfinite(V.d_shuf[lr]).all().item())
    cos = torch.cosine_similarity(V.d_shuf, V.s, dim=-1)[lr].abs().max().item()
    ck("d_shuf ортогонален s на рабочих слоях", cos < 1e-3, f"макс. |cos| {cos:.1e}")
    ns = V.norm_s
    worst = max((v[lr].norm(dim=-1, keepdim=True) / ns[lr] - 1).abs().max().item() for lst in vecs.values() for _, v, _ in lst)
    ck("все векторы условий имеют норму ||s|| на рабочих слоях", worst < 1e-3, f"макс. отклонение {worst:.1e}")
    ck("rand_unit детерминирован по seed", torch.equal(rand_unit(0, V.s, ns), rand_unit(0, V.s, ns)))
    ck("random0 и random1 различаются", not torch.equal(rand_unit(0, V.s, ns), rand_unit(1, V.s, ns)))
    d0 = dict((c, v) for c, v, _ in vecs[TOOLS[0]["id"]])
    ck("steer_neg = -steer на рабочих слоях", torch.allclose(d0["steer_neg"][lr], -d0["steer"][lr]))
    fl = next(t for t in TOOLS if t["domain"] == "flight" and t["order"] == "do" and t["layout"] == "date_mid")
    dd_ = dict((c, v) for c, v, _ in vecs[fl["id"]])
    e = V.e_lay["date_mid"][lr]
    ref = ns[lr] * e * -1 / e.norm(dim=-1, keepdim=True)
    ck("steer_pos для порядка do берётся со знаком минус", torch.allclose(dd_["steer_pos"][lr], ref, atol=1e-4 * ns[lr].max().item()))
    ck("у рейсов 6 векторов семейства steer, у остальных 5",
       all(len([c for c, _, _ in vecs[t["id"]] if c.startswith("steer") and c != "steer_neg"]) == (6 if t["domain"] == "flight" else 5)
           for t in TOOLS))
    ck("steer_dom только у message/period, steer_layout и steer_pos только у рейсов",
       all(("steer_dom" in {c for c, _, _ in vecs[t["id"]]}) == (t["domain"] != "flight") and
           ("steer_layout" in {c for c, _, _ in vecs[t["id"]]}) == (t["domain"] == "flight") for t in TOOLS))


def n_specs(cfg, vecs, dom):
    """Число пар (условие, сила) для типа задач dom."""
    t = next(t for t in TOOLS if t["domain"] == dom)
    return sum(len(al) for _, _, al in vecs[t["id"]])


def check_eval(ck, lm, cfg, vecs, pool):
    """Прогоняет eval_example для каждого типа задач и проверяет вид результата."""
    rnd = random.Random(123)
    for dom in ("flight", "message", "period"):
        t = next(t for t in TOOLS if t["domain"] == dom)
        tpl = next(x for x in TEMPLATES if x["domain"] == dom)
        O, D, X = rnd.sample(pool[dom]["test"], 3)
        rows = eval_example(lm, cfg, vecs, t, tpl, O, D, X, "second")
        S = n_specs(cfg, vecs, dom)
        ck(f"eval_example[{dom}]: число строк = 1 + слои·(условия + patch)", len(rows) == 1 + len(cfg.LAYER_RANGE) * (S + 1),
           f"{len(rows)} из {1 + len(cfg.LAYER_RANGE) * (S + 1)}")
        ck(f"eval_example[{dom}]: первая строка — базовая (layer=-1, cond=none)", rows[0]["layer"] == -1 and rows[0]["cond"] == "none")
        ck(f"eval_example[{dom}]: есть patch на каждом слое", sorted(r["layer"] for r in rows if r["cond"] == "patch") == list(cfg.LAYER_RANGE))
        ck(f"eval_example[{dom}]: delta конечны", all(pd.notna(r["delta"]) and abs(r["delta"]) < 1e6 for r in rows))
        ck(f"eval_example[{dom}]: top1 — не более одного из T/S/X", all(r["top1_is_T"] + r["top1_is_S"] + r["top1_is_X"] <= 1 for r in rows))


def check_outputs(ck, cfg, vecs, df, dd, summary):
    """Проверяет созданные файлы и число строк raw.csv."""
    out = cfg.OUT_DIR
    for f in ["run.log", "config.json", "vectors.pt", "vector_geometry.csv", "raw.csv"]:
        ck(f"файл {f} создан", os.path.exists(f"{out}/{f}"))
    exp = sum(n * (1 + len(cfg.LAYER_RANGE) * (n_specs(cfg, vecs, d) + 1))
              for d, n in [("flight", cfg.N_TEST), ("message", cfg.N_TEST_DOM), ("period", cfg.N_TEST_DOM)])
    ck("raw.csv: число строк сходится с ожидаемым", len(df) == exp and len(pd.read_csv(f"{out}/raw.csv")) == exp, f"{len(df)} из {exp}")
    ck.warn("после фильтра base_ok остались примеры (иначе анализ на реальных данных пропущен)", dd is not None)
    if dd is not None:
        check_analysis(ck, cfg, summary, "реальные данные")
    check_analysis_forced(ck, cfg, df)


def check_analysis(ck, cfg, summary, label):
    """Проверяет таблицы и графики анализа."""
    out = cfg.OUT_DIR
    ck(f"анализ[{label}]: доли pT+pS+pX+other = 1", ((summary.pT + summary.pS + summary.pX + summary.other) - 1).abs().max() < 1e-9)
    ck(f"анализ[{label}]: доли в [0, 1]", summary[["pT", "pS", "pX", "other"]].stack().between(0, 1).all())
    for f in ["summary.csv", "base_rates.csv", f"cell_stats_L{cfg.L0}_A{cfg.A0}.csv", f"mcnemar_L{cfg.L0}_A{cfg.A0}.csv",
              f"where_goes_L{cfg.L0}_A{cfg.A0}.csv"]:
        ck(f"анализ[{label}]: файл {f} создан", os.path.exists(f"{out}/{f}"))
    ck(f"анализ[{label}]: построен хотя бы один график", any(f.endswith(".png") for f in os.listdir(out)))


def check_analysis_forced(ck, cfg, df):
    """Прогоняет анализ на копии raw.csv, где все примеры считаются верными: проверка не зависит от качества модели."""
    df2 = df.copy()
    df2["base_ok"] = True
    df2["first_ok"] = df2.first_ok.map(lambda v: None if v is None or pd.isna(v) else True)
    sub = replace(cfg, OUT_DIR=f"{cfg.OUT_DIR}/analysis_forced")
    os.makedirs(sub.OUT_DIR, exist_ok=True)
    res = analysis.analyze(sub, df2, ck.log)
    ck("анализ[forced]: analyze вернул данные", res is not None)
    if res is None:
        return
    dd, summary = res
    plots.make_plots(sub, summary, ck.log)
    analysis.stats(sub, dd, ck.log)
    check_analysis(ck, sub, summary, "forced")
