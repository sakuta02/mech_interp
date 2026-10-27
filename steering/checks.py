"""Проверки режима --test для опыта: на каждом этапе проверяют формы, нормы, работу хуков и итоговые файлы
и пишут в лог OK, FAIL или WARN. Глобальный random не трогают: используют локальный Random."""
import os, random
from dataclasses import replace
import pandas as pd, torch

import analysis, plots
from data import N_ARGS, TOOLS, TEMPLATES, TRANSITIONS, SLOTS, TYPE_CANDIDATES, sample_vals
from evaluate import eval_example
from model import RESID, steer
from prompts import call_ids, render_user, probe_format
from vectors import rand_unit, pooled


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
    ck("5 инструментов по 5 аргументов", len(TOOLS) == 5 and all(len(t["args"]) == N_ARGS for t in TOOLS))
    ck("переходы: 1→3, 3→5, 5→3, 3→1", TRANSITIONS == [(1, 3), (3, 5), (5, 3), (3, 1)])
    ck("слоты для векторов: 1, 3, 5", SLOTS == [1, 3, 5])
    ck("внутри инструмента типы всех аргументов различны", all(len({a["type"] for a in t["args"]}) == N_ARGS for t in TOOLS))
    ck("внутри инструмента ключи аргументов различны", all(len({a["key"] for a in t["args"]}) == N_ARGS for t in TOOLS))
    t135 = [t["args"][k - 1]["type"] for t in TOOLS for k in SLOTS]
    ck("на позициях 1, 3, 5 стоят 15 разных типов", len(set(t135)) == 15, f"{len(set(t135))}")
    ck("есть инструмент, где имя и город стоят на слотах 1, 3, 5 (имя ← пункт назначения)",
       any({"name", "city"} <= {t["args"][k - 1]["type"] for k in SLOTS} for t in TOOLS))
    ck("id инструментов уникальны", len({t["id"] for t in TOOLS}) == len(TOOLS))
    ck("шаблоны: есть calib и test, fwd и rev", {x["split"] for x in TEMPLATES} == {"calib", "test"}
       and {x["rev"] for x in TEMPLATES} == {True, False})
    used = {a["type"] for t in TOOLS for a in t["args"]}
    ck("пулы построены для всех используемых типов", used <= set(pool) <= set(TYPE_CANDIDATES))
    flat = [v for ty in pool for k in ("calib", "test") for v in pool[ty][k]]
    toks = [lm.enc(v)[0] for v in flat]
    ck("все значения пулов однотокенные", all(len(lm.enc(v)) == 1 for v in flat))
    ck("токены значений не повторяются ни внутри типа, ни между типами", len(set(toks)) == len(toks), f"{len(toks) - len(set(toks))} повторов")
    ck("калибровочный и тестовый пулы каждого типа не пересекаются", all(not set(pool[ty]["calib"]) & set(pool[ty]["test"]) for ty in pool))
    low = min(len(pool[ty][k]) for ty in pool for k in ("calib", "test"))
    ck("в каждом пуле не меньше 4 значений", low >= 4, f"минимум {low}")


def check_prompts(ck, lm, pool):
    """Проверяет сборку промптов: позиции слотов, обрезку upto и порядок значений в шаблонах."""
    bad_pos, bad_cut, n = 0, 0, 0
    for t in TOOLS:
        for tpl in TEMPLATES:
            vals = {k: pool[a["type"]]["calib"][0] for k, a in enumerate(t["args"], 1)}
            ids, pos = call_ids(lm, t, tpl, vals)
            n += 1
            bad_pos += any(ids[pos[k] + 1] != lm.enc(vals[k])[0] for k in range(1, N_ARGS + 1))
            for k in range(1, N_ARGS + 1):
                cut, pc = call_ids(lm, t, tpl, vals, upto=k)
                bad_cut += cut != ids[:pos[k] + 1] or pc[k] != pos[k] or len(cut) != pos[k] + 1
    ck(f"call_ids: токен после pos[k] — значение аргумента k ({n} комбинаций)", bad_pos == 0, f"ошибок {bad_pos}")
    ck("call_ids(upto=k) совпадает с началом полного вызова", bad_cut == 0, f"ошибок {bad_cut}")
    t = TOOLS[0]
    vals = {k: pool[a["type"]]["calib"][0] for k, a in enumerate(t["args"], 1)}
    fwd = render_user(t, next(x for x in TEMPLATES if x["id"] == "kv_fwd"), vals)
    rev = render_user(t, next(x for x in TEMPLATES if x["id"] == "kv_rev"), vals)
    first, last = t["args"][0]["label"], t["args"][-1]["label"]
    ck("шаблон fwd перечисляет аргументы 1..5, rev — 5..1", fwd.index(first) < fwd.index(last) and rev.index(first) > rev.index(last))
    ck("в сообщении пользователя есть все пять значений", all(v in fwd for v in vals.values()))


def check_model(ck, lm, cfg, pool):
    """Проверяет чтение активаций хуком и правку активаций хуками на загруженной модели."""
    t, tpl = TOOLS[0], TEMPLATES[0]
    vals = sample_vals(t, pool, "calib", random.Random(1))
    ids, pos = call_ids(lm, t, tpl, vals)
    logits, st = lm.run_states(lm.tensor(ids), {k: pos[k] for k in SLOTS})
    ck("run_states: логиты [T, V]", logits.shape[0] == len(ids) and logits.ndim == 2, f"{tuple(logits.shape)}")
    ck("run_states: состояния [слои, d] для слотов 1, 3, 5", set(st) == set(SLOTS) and all(v.shape[0] == lm.n_layers and v.ndim == 2 for v in st.values()),
       f"{ {k: tuple(v.shape) for k, v in st.items()} }")
    ck("run_states: нет NaN/inf", all(torch.isfinite(v).all().item() for v in st.values()))
    _, cache = lm.model.run_with_cache(lm.tensor(ids), names_filter=[RESID(l) for l in range(lm.n_layers)])
    ck("читающий хук даёт те же активации, что run_with_cache (бит в бит)",
       all(torch.equal(st[k], torch.stack([cache[RESID(l)][0, pos[k]] for l in range(lm.n_layers)]).float()) for k in SLOTS))
    ck("run_states(logits=False) не считает логиты", lm.run_states(lm.tensor(ids), {1: 0}, logits=False)[0] is None)

    ids3, _ = call_ids(lm, t, tpl, vals, upto=3)
    x = lm.tensor(ids3)
    base = lm.fwd_last(x)
    l = cfg.L0
    d = lambda a, b: (a - b).abs().max().item()
    ck("fwd_last: форма [1, V]", base.ndim == 2 and base.shape[0] == 1, f"{tuple(base.shape)}")
    dl = d(lm.model(x)[0, -1][None].float(), base)
    ck("model(x) и fwd_last согласованы по последней позиции", dl < 0.5, f"макс. разница логитов {dl:.3f}")
    stx = lm.run_states(x, {"p": x.shape[1] - 1}, logits=False)[1]["p"]
    zero = torch.zeros(1, stx.shape[-1], device=lm.device)
    ck("хук steer(adds=0) не меняет логиты", d(lm.fwd_last(x, [(RESID(l), steer(adds=zero))]), base) < 1e-3)
    ck("хук steer(adds=вектор) меняет логиты", d(lm.fwd_last(x, [(RESID(l), steer(adds=stx[l][None] * 0.5))]), base) > 0)
    ck("хуки снимаются после прогона", d(lm.fwd_last(x), base) < 1e-3)
    ck("подмена собственной активацией (repl) не меняет логиты", d(lm.fwd_last(x, [(RESID(l), steer(repl=stx[l]))]), base) < 1e-3)
    lg3 = lm.fwd_last(x.repeat(3, 1))
    worst = max(d(lg3[i:i + 1], base) for i in range(3))
    ck("батч из одинаковых строк даёт те же логиты, что одна строка", worst < 1.0, f"макс. разница {worst:.3f}")
    ck("generate отрабатывает и возвращает строку", isinstance(probe_format(lm, lambda *a: None), str))


def check_vectors(ck, cfg, H, V, vecs):
    """Проверяет формы, нормы и ортогональность векторов и контролей.
    Слой 0 исключён: там resid_pre — эмбеддинг одного токена перед всеми слотами, native = 0 и нормировка даёт NaN; стиринг идёт только по LAYER_RANGE."""
    lr = list(cfg.LAYER_RANGE)
    ck("калибровка: для каждого инструмента состояния слотов 1, 3, 5 одинаковой формы [kept, слои, d]",
       all(set(h) == set(SLOTS) and len({tuple(v.shape) for v in h.values()}) == 1 and next(iter(h.values())).ndim == 3 for h in H.values()))
    ck("векторы заведены для каждой пары (инструмент, переход)", len(V.native) == len(V.foreign) == len(V.shuf) == len(TOOLS) * len(TRANSITIONS))
    ck("векторы без NaN/inf на рабочих слоях", all(torch.isfinite(v[lr]).all().item() for D in (V.native, V.foreign, V.shuf) for v in D.values()))
    ck("обратный переход равен прямому с минусом (5→3 = −(3→5), 3→1 = −(1→3))",
       all(torch.allclose(V.native[(t["id"], (5, 3))], -V.native[(t["id"], (3, 5))]) and
           torch.allclose(V.native[(t["id"], (3, 1))], -V.native[(t["id"], (1, 3))]) for t in TOOLS))
    t0 = TOOLS[0]["id"]
    ck("native = среднее состояния слота d минус слота o по примерам инструмента", torch.equal(pooled(H, [t0], 1, 3), V.native[(t0, (1, 3))]))
    others = [t["id"] for t in TOOLS if t["id"] != t0]
    ck("foreign = тот же переход по остальным инструментам, без своего", torch.equal(pooled(H, others, 3, 5), V.foreign[(t0, (3, 5))]))
    ck("foreign не совпадает с native", not torch.equal(V.foreign[(t0, (3, 5))], V.native[(t0, (3, 5))]))
    cos = max(torch.cosine_similarity(V.shuf[k], V.native[k], dim=-1)[lr].abs().max().item() for k in V.shuf)
    ck("shuffled ортогонален своему native на рабочих слоях", cos < 1e-3, f"макс. |cos| {cos:.1e}")
    worst = max((v[lr].norm(dim=-1) / V.native[k][lr].norm(dim=-1) - 1).abs().max().item()
                for k, lst in vecs.items() for _, _, v, _ in lst)
    ck("все векторы условий имеют норму своего native на рабочих слоях", worst < 1e-3, f"макс. отклонение {worst:.1e}")
    ck("число условий на пару (инструмент, переход) = 6: свой, неродной и 4 контроля", all(len(lst) == 6 for lst in vecs.values()))
    ck("имена условий: steer_tool_{o}_{d}, steer_foreign_{o}_{d} и контроли",
       all([c for c, _, _, _ in vecs[(t["id"], tr)]] == [f"steer_tool_{tr[0]}_{tr[1]}", f"steer_foreign_{tr[0]}_{tr[1]}",
                                                         "steer_neg", "shuffled", "random0", "random1"] for t in TOOLS for tr in TRANSITIONS))
    r0 = V.native[(t0, (1, 3))]
    ck("rand_unit детерминирован по seed", torch.equal(rand_unit(0, r0), rand_unit(0, r0)))
    ck("random0 и random1 различаются", not torch.equal(rand_unit(0, r0), rand_unit(1, r0)))
    d0 = {c: v for c, _, v, _ in vecs[(t0, (1, 3))]}
    ck("steer_neg = −steer_tool на рабочих слоях", torch.allclose(d0["steer_neg"][lr], -d0["steer_tool_1_3"][lr]))


def n_specs(vecs, key):
    """Число пар (условие, сила) для ключа (инструмент, переход)."""
    return sum(len(al) for _, _, _, al in vecs[key])


def check_eval(ck, lm, cfg, vecs, pool):
    """Прогоняет eval_example для каждого инструмента и всех переходов и проверяет вид результата."""
    rnd = random.Random(123)
    for i, t in enumerate(TOOLS):
        for tr in (TRANSITIONS if i == 0 else [TRANSITIONS[i % len(TRANSITIONS)]]):
            tpl = TEMPLATES[i % len(TEMPLATES)]
            vals = sample_vals(t, pool, "test", rnd)
            rows = eval_example(lm, cfg, vecs, t, tpl, tr, vals)
            name = f"eval_example[{t['id']} {tr[0]}→{tr[1]}]"
            exp = 1 + len(cfg.LAYER_RANGE) * (n_specs(vecs, (t["id"], tr)) + 1)
            ck(f"{name}: число строк = 1 + слои·(условия + patch)", len(rows) == exp, f"{len(rows)} из {exp}")
            ck(f"{name}: первая строка — базовая (layer=-1, cond=none)", rows[0]["layer"] == -1 and rows[0]["cond"] == "none")
            ck(f"{name}: patch на каждом слое", sorted(r["layer"] for r in rows if r["kind"] == "patch") == list(cfg.LAYER_RANGE))
            ck(f"{name}: условия steer_tool_{tr[0]}_{tr[1]} и steer_foreign_{tr[0]}_{tr[1]} присутствуют",
               {f"steer_tool_{tr[0]}_{tr[1]}", f"steer_foreign_{tr[0]}_{tr[1]}"} <= {r["cond"] for r in rows})
            ck(f"{name}: S и T — значения аргументов o и d", rows[0]["S"] == vals[tr[0]] and rows[0]["T"] == vals[tr[1]])
            ck(f"{name}: delta конечны", all(pd.notna(r["delta"]) and abs(r["delta"]) < 1e6 for r in rows))
            ck(f"{name}: top1 — не более одного из T/S/R", all(r["top1_is_T"] + r["top1_is_S"] + r["top1_is_R"] <= 1 for r in rows))
            if tr[0] == 1:
                ck(f"{name}: для первого аргумента контекст верен без проверки (ctx_ok)", rows[0]["ctx_ok"] is True)


def check_outputs(ck, cfg, vecs, df, dd, summary):
    """Проверяет созданные файлы и число строк raw.csv, затем анализ на реальных данных и на принудительно верных."""
    out = cfg.OUT_DIR
    for f in ["run.log", "config.json", "vectors.pt", "vector_geometry.csv", "raw.csv"]:
        ck(f"файл {f} создан", os.path.exists(f"{out}/{f}"))
    exp = cfg.N_TEST * (1 + len(cfg.LAYER_RANGE) * (n_specs(vecs, next(iter(vecs))) + 1))
    ck("raw.csv: число строк сходится с ожидаемым", len(df) == exp and len(pd.read_csv(f"{out}/raw.csv")) == exp, f"{len(df)} из {exp}")
    ck.warn("после фильтра base_ok и ctx_ok остались примеры (иначе анализ на реальных данных пропущен)", dd is not None)
    if dd is not None:
        check_analysis(ck, cfg, summary, "реальные данные")
    check_analysis_forced(ck, cfg, df)


def check_analysis(ck, cfg, summary, label):
    """Проверяет таблицы и графики анализа."""
    out = cfg.OUT_DIR
    ck(f"анализ[{label}]: доли pT+pS+pR+other = 1", ((summary.pT + summary.pS + summary.pR + summary.other) - 1).abs().max() < 1e-9)
    ck(f"анализ[{label}]: доли в [0, 1]", summary[["pT", "pS", "pR", "other"]].stack().between(0, 1).all())
    ck(f"анализ[{label}]: в summary есть сводные строки ALL", "ALL" in set(summary.tool_id))
    for f in ["summary.csv", "base_rates.csv", f"cell_stats_L{cfg.L0}_A{cfg.A0}.csv", f"gap_L{cfg.L0}_A{cfg.A0}.csv",
              f"mcnemar_L{cfg.L0}_A{cfg.A0}.csv", f"where_goes_L{cfg.L0}_A{cfg.A0}.csv"]:
        ck(f"анализ[{label}]: файл {f} создан", os.path.exists(f"{out}/{f}"))
    pngs = [f for f in os.listdir(out) if f.endswith(".png")]
    ck(f"анализ[{label}]: построены 3 графика (по слоям, по силе, неродные 3→5 и 5→3)", len(pngs) == 3, f"{pngs}")


def check_analysis_forced(ck, cfg, df):
    """Прогоняет анализ на копии raw.csv, где все примеры считаются верными: проверка не зависит от качества модели."""
    df2 = df.copy()
    df2["base_ok"] = True
    df2["ctx_ok"] = True
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
