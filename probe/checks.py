"""Проверки режима --test: данные, промпты и позиции токенов, хуки, метки проб, сами пробы на синтетических сигналах
и сквозной прогон на подложенном сигнале. В лог пишутся OK, FAIL или WARN. Глобальный random не трогают."""
import os, random
import numpy as np, pandas as pd, torch

from data import FIT_GROUPS, GROUP_SPLITS, NAMES, DESCS, TEMPLATES, N_SLOTS, ROLES, OTHER, make_spec
import patching, probes
from prompts import call_ids, render_user


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


def check_data(ck, lm, pools):
    """Названия, описания, шаблоны, пулы и устройство примеров всех групп."""
    tl = pools["tlen"]
    allk = [k for r in NAMES.values() for s in r.values() for k in s]
    ck("названия полей уникальны", len(set(allk)) == len(allk))
    for sp in ("train", "test"):
        common = {tl[k] for k in NAMES["src"][sp]} & {tl[k] for k in NAMES["dst"][sp]}
        ck(f"в split {sp} есть названия src и dst одной длины в токенах", len(common) > 0, f"длины {sorted(common)}")
    ck("есть train и test описания и шаблоны", all(DESCS[r][s] for r in ROLES for s in ("train", "test"))
       and {t["split"] for t in TEMPLATES} == {"train", "test"})
    ck("шаблоны: и prefix, и postfix в каждом split", all({t["pos"] for t in TEMPLATES if t["split"] == s} == {"prefix", "postfix"}
                                                       for s in ("train", "test")))
    v = pools["vals"]
    flat = [x for ty in v for s in v[ty] for x in v[ty][s]]
    toks = [lm.enc(x)[0] for x in flat]
    ck("значения однотокенные и не повторяются между типами и split", all(len(lm.enc(x)) == 1 for x in flat) and len(set(toks)) == len(toks))
    rng, bad = random.Random(1), {}
    for g in GROUP_SPLITS:
        for _ in range(60):
            sp = make_spec(rng, pools, g)
            f, s = sp["fields"], GROUP_SPLITS[g]
            keys = [x["key"] for x in f]
            src, dst = next(x for x in f if x["role"] == "src"), next(x for x in f if x["role"] == "dst")
            tok = [lm.enc(x["written"]) for x in f]
            conds = {"роли: src, dst, f1, f2": sorted(x["role"] for x in f) == ["dst", "f1", "f2", "src"],
                     "ключи полей различны": len(set(keys)) == 4,
                     "названия src и dst одной длины": tl[src["key"]] == tl[dst["key"]],
                     "значения однотокенные и различны (в swap город записан дважды, различность по value)":
                         all(len(t) == 1 for t in tok) and len({lm.enc(x["value"])[0] for x in f}) == 4,
                     "города из нужного split": sp["src_city"] in v["city"][s["city"]] and sp["dst_city"] in v["city"][s["city"]],
                     "названия из нужного split": src["key"] in NAMES["src"][s["name"]] and dst["key"] in NAMES["dst"][s["name"]],
                     "описания из нужного split (у conflict — из пула другой роли)":
                         all(x["desc"] in DESCS[x["desc_role"]][s["desc"]] for x in (src, dst)),
                     "шаблон из нужного split": next(t for t in TEMPLATES if t["id"] == sp["tpl"])["split"] == s["tpl"],
                     "first согласован с шаблоном": next(t for t in TEMPLATES if t["id"] == sp["tpl"])["first"] in (None, sp["first"]),
                     "swap: первое поле городское, записан город другой роли":
                         g != "swap" or (f[0]["role"] in ROLES and f[0]["written"] == sp["dst_city" if f[0]["role"] == "src" else "src_city"]),
                     "не swap: записано золото": g == "swap" or all(x["written"] == x["value"] for x in f),
                     "postfix: только postfix-шаблоны": g != "postfix" or sp["tpl_pos"] == "postfix",
                     "conflict: описание городского поля взято у другой роли, иначе у своей":
                         all(x["desc"] in DESCS[OTHER[x["role"]] if g == "conflict" else x["role"]][s["desc"]]
                             for x in f if x["role"] in ROLES)}
            for k, c in conds.items():
                bad[k] = bad.get(k, 0) + (not c)
    for k, n in bad.items():
        ck(f"данные: {k} (60 примеров × {len(GROUP_SPLITS)} групп)", n == 0, f"ошибок {n}")


def check_prompts(ck, lm, pools):
    """Позиции слотов и городов, порядок ключей схемы, интервалы городов в сообщении."""
    rng, bad = random.Random(2), dict(pos=0, city=0, order=0, span=0, marker=0)
    n = 0
    for t in TEMPLATES:
        for _ in range(20):
            g = "iid" if t["split"] == "train" else "ood_tpl"
            sp = make_spec(rng, pools, g)
            while sp["tpl"] != t["id"]:
                sp = make_spec(rng, pools, g)
            ids, pos, cpos = call_ids(lm, sp)
            text, spans = render_user(sp)
            n += 1
            bad["pos"] += any(ids[pos[k] + 1] != lm.enc(sp["fields"][k - 1]["written"])[0] for k in pos)
            bad["span"] += any(text[a:b] != sp[f"{r}_city"] for r, (a, b) in spans.items())
            bad["city"] += any(not sp[f"{r}_city"].endswith(lm.tok.decode([ids[cpos[r]]]).strip()) or
                               not lm.tok.decode([ids[cpos[r]]]).strip() for r in ROLES)
            s = lm.tok.decode(ids)
            idx = [s.index(f'"{x["key"]}": {{') for x in sp["fields"]]
            bad["order"] += idx != sorted(idx)
            if sp["tpl_pos"] == "prefix":
                before = lm.tok.decode(ids[max(0, cpos["src"] - 6):cpos["src"] + 1])
                bad["marker"] += not any(w in before for w in ("from", "departing", "starting in"))
    ck(f"call_ids: токен после pos[k] — записанное значение поля k ({n} примеров)", bad["pos"] == 0, f"ошибок {bad['pos']}")
    ck("интервалы городов в сообщении совпадают с городами", bad["span"] == 0)
    ck("последний токен города на cpos — конец названия города", bad["city"] == 0, f"ошибок {bad['city']}")
    ck("ключи схемы в промпте идут в порядке слотов", bad["order"] == 0, f"ошибок {bad['order']}")
    ck("prefix-шаблоны: перед городом src стоит маркер роли", bad["marker"] == 0, f"ошибок {bad['marker']}")


def check_model(ck, lm, pools):
    """Состояния run_states совпадают с run_with_cache; форма результата."""
    sp = make_spec(random.Random(3), pools, "iid")
    ids, pos, cpos = call_ids(lm, sp)
    sites = {f"slot{k}": pos[k] for k in pos}
    logits, st = lm.run_states(lm.tensor(ids), sites)
    ck("run_states: форма [слои, d] на позицию", all(v.shape == (lm.n_layers, lm.model.cfg.d_model) for v in st.values()))
    ck("run_states: логиты [T, V]", logits.shape[0] == len(ids))
    _, cache = lm.model.run_with_cache(lm.tensor(ids), names_filter=[f"blocks.{l}.hook_resid_pre" for l in (0, 10, lm.n_layers - 1)])
    for l in (0, 10, lm.n_layers - 1):
        a, b = st["slot2"][l], cache[f"blocks.{l}.hook_resid_pre"][0, pos[2]].float()
        ck(f"хуки: resid_pre слоя {l} совпадает с run_with_cache", (a - b).norm() / b.norm() < 1e-2, f"{float((a - b).norm() / b.norm()):.1e}")
    ck("состояния конечны", all(torch.isfinite(v).all() for v in st.values()))


def check_rows(ck, df, lexm, qs, cfg):
    """Метки и вопросы: независимость Q4, согласованность splits, отсутствие утечек между наборами."""
    sl = df[(df.site < N_SLOTS) & (df.k == 2) & (df.group != "swap")]
    st = set(zip(sl.u_src, sl.u_dst))
    ck("Q4: на слоте 2 состояния (нет, только src, только dst) все встречаются и (1,1) невозможно", st == {(0, 0), (1, 0), (0, 1)}, f"{sorted(st)}")
    ck("Q4: метки u_src и u_dst не зеркальны", abs(np.corrcoef(sl.u_src, sl.u_dst)[0, 1]) < 0.9, f"corr {np.corrcoef(sl.u_src, sl.u_dst)[0, 1]:.2f}")
    sw = df[(df.group == "swap") & (df.k == 2)]
    ck("swap: поле-метка и город-метка расходятся на слоте 2 в городских первых полях", len(sw) > 0 and
       ((sw.u_src != sw.uc_src) & (sw.u_dst != sw.uc_dst)).all())
    ck("в swap нет проверки правильности и нет pointer-меток", df[df.group == "swap"].correct.all() and (df[df.group == "swap"].ptr_role == -1).all())
    ok = df[(df.site < N_SLOTS) & (df.ptr_role >= 0) & df.correct & df.cur_city & (df.group != "conflict")]
    ck("на верных примерах pointer по роли совпадает с ролью слота", (ok.ptr_role == ok.y_role).all(), f"{len(ok)} строк")
    lk = df[(df.site >= N_SLOTS) & (df.tpl_pos == "postfix") & (df.y_role == df.first_id)]
    ck("Q0_leak: у первого города postfix-сообщения роли src и dst равновероятны (роль неизвестна по контексту)", len(lk) > 0 and 0.3 < lk.y_role.mean() < 0.7, f"доля dst {lk.y_role.mean():.2f}")
    sl_all = df[(df.site < N_SLOTS) & (df.group != "swap")]
    ck("dist_r > 0 ровно там, где поле роли r уже записано", all(((sl_all[f"dist_{r}"] > 0) == (sl_all[f"u_{r}"] == 1)).all() for r in ROLES))
    e = df[df.err_slot]
    ck("ошибочный слот: модель выбрала город другой роли", len(e) == 0 or (e.ptr_role != e.y_role).all(), f"{len(e)} слотов")
    cf = df[(df.group == "conflict") & df.cur_city & (df.site < N_SLOTS)]
    ck("conflict: роль по описанию противоположна роли по названию", len(cf) > 0 and (cf.desc_role != cf.y_role).all())
    nc = df[(df.group != "conflict") & (df.site < N_SLOTS)]
    ck("вне conflict роль по описанию совпадает с ролью поля", (nc.desc_role == nc.y_role).all())
    ck("postfix-группа содержит только postfix-шаблоны", (df[df.group == "postfix"].tpl_pos == "postfix").all())
    ck("метки y_ord = k − 1 на слотах", (df[df.site < N_SLOTS].y_ord == df[df.site < N_SLOTS].k - 1).all())
    fs = df.groupby("ex").fit_split.nunique()
    ck("каждый пример целиком в одном split", (fs == 1).all())
    ck("fit/val/test есть только у групп iid и postfix", set(df[~df.group.isin(FIT_GROUPS)].fit_split) == {"-"}
       and all({"fit", "val", "test"} <= set(df[df.group == g].fit_split) for g in FIT_GROUPS))
    ck("лексическая матрица согласована со строками", lexm.shape[0] == len(df) and lexm[(df.site >= N_SLOTS).values].sum() == 0)
    for q in qs:
        n_fit, n_val = int(q["fit"].sum()), int(q["val"].sum())
        two = df.loc[q["fit"], q["y"]].nunique() >= 2
        (ck if n_fit >= 100 else ck.warn)(f"{q['name']}: fit не пуст и в нём не меньше двух классов", n_fit > 0 and two, f"fit {n_fit}, val {n_val}")
        ck(f"{q['name']}: fit и оценочные наборы не пересекаются", all(not (q["fit"] & m).any() for m, _ in q["evals"].values()))
        ck(f"{q['name']}: метки в пределах числа классов", df.loc[q["base"], q["y"]].between(0, q["ncls"] - 1).all())
        empty = [k for k, (m, _) in q["evals"].items() if int(m.sum()) == 0]
        ck.warn(f"{q['name']}: все оценочные наборы непусты", not empty, f"пустые: {empty}")


def check_probes(ck, device, cfg):
    """Пробы на синтетике: подложенный сигнал читается, перемешанные метки дают шанс, метрики и бутстреп корректны."""
    g = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    N, d = 400, 64
    y = torch.randint(0, 3, (N,), generator=g)
    D = torch.randn(3, d, generator=g)
    X = torch.randn(N, d, generator=g) + 1.5 * D[y]
    X, yd = X.to(device), y.to(device)
    fit, val, te = np.arange(0, 240), np.arange(240, 320), np.arange(320, 400)
    grp = np.arange(N)
    cfg2 = cfg
    res, _ = probes.probe_suite(X, yd, fit, val, {"t": (te, None)}, grp, 3, cfg2, rng)
    r = {x["probe"]: x for x in res}
    ck("пробы: логрег читает подложенный сигнал (3 класса)", r["logreg"]["bacc"] > 0.9, f"{r['logreg']['bacc']:.2f}")
    ck("пробы: центроид читает подложенный сигнал", r["dim"]["bacc"] > 0.9, f"{r['dim']['bacc']:.2f}")
    ck("пробы: перемешанные метки дают шанс", r["shuf0"]["bacc"] < 0.55, f"{r['shuf0']['bacc']:.2f}")
    ck("пробы: интервал накрывает оценку", r["logreg"]["lo"] <= r["logreg"]["bacc"] <= r["logreg"]["hi"])
    yb = (y == 0).long()
    res2, cos = probes.probe_suite(X, yb.to(device), fit, val, {"t": (te, None)}, grp, 2, cfg2, rng)
    ck("пробы: AUROC близок к 1 на разделимых классах", res2[0]["auroc"] > 0.95, f"{res2[0]['auroc']:.2f}")
    ck("пробы: cos между направлениями логрега и центроида положителен", cos is not None and cos > 0, f"{cos:.2f}")
    s = np.r_[np.zeros(50), np.ones(50)]
    ck("auroc: идеальное разделение даёт 1, разворот 0", probes.auroc(s.astype(int), s) == 1 and probes.auroc(s.astype(int), -s) == 0)
    ck("auroc: случайные оценки дают около 0.5", abs(probes.auroc(s.astype(int), rng.normal(size=100)) - 0.5) < 0.2)
    ck("bacc: учитывает только присутствующие классы", probes.bacc(np.array([0, 0, 1]), np.array([0, 0, 1]), 3) == 1)
    lo, hi = probes.boot_ci(np.array([0, 1] * 50), np.array([0, 1] * 40 + [1, 0] * 10), np.arange(100) // 2, 2, 100, rng)
    ck("boot_ci: границы в [0, 1] и lo ≤ hi", 0 <= lo <= hi <= 1, f"{lo:.2f}..{hi:.2f}")


def check_planted(ck, device, cfg, df, lexm, qs, H_shape):
    """Сквозная проверка на подложенном сигнале: шум в H, метка Q1 заложена на слое 10, метка Q4 (по полю) на слое 20.
    Проба должна читать её ровно на этих слоях и на всех оценочных наборах; swap_city (метка по городу) читаться не должна."""
    L, N, S, d = H_shape
    g = torch.Generator().manual_seed(1)
    H = torch.randn(L, N, S, d, generator=g).half()
    sl = df[df.site < N_SLOTS]
    D1, D4 = torch.randn(3, d, generator=g), torch.randn(2, d, generator=g)
    ex, site = torch.as_tensor(sl.ex.values.copy()), torch.as_tensor(sl.site.values.copy())
    H[10][ex, site] += (0.3 * D1[torch.as_tensor(sl.y_role.values.copy())]).half()
    r2 = df[df.k == 2]
    ex2, site2 = torch.as_tensor(r2.ex.values.copy()), torch.as_tensor(r2.site.values.copy())
    H[20][ex2, site2] += (0.3 * D4[torch.as_tensor(r2.u_src.values.copy())]).half()
    rng = np.random.default_rng(0)
    layers = [l for l in (0, 10, 20) if l < L]
    for name, lay, sets in [("Q1_role", 10, ["iid_test", "ood_name", "ood_tpl", "ood_all"]),
                            ("Q4_used_src", 20, ["iid_test", "ood_all", "swap_field"])]:
        q = next(x for x in qs if x["name"] == name)
        res, _ = probes.run_question(q, df, H, lexm, layers, cfg, rng, device, lambda *a: None)
        res = pd.DataFrame(res)
        for s in sets:
            x = res[(res.probe == "logreg") & (res.set == s)].set_index("layer").bacc
            if lay in x.index and not np.isnan(x[lay]):
                ck(f"подложенный сигнал {name}: читается на слое {lay}, набор {s}", x[lay] > 0.85, f"{x[lay]:.2f}")
            others = [v for l, v in x.items() if l != lay and not np.isnan(v)]
            ck.warn(f"подложенный сигнал {name}: на других слоях шанс, набор {s}", all(v < 0.7 for v in others), f"{[round(v, 2) for v in others]}")
    q = next(x for x in qs if x["name"] == "Q4_used_src")
    res, _ = probes.run_question(q, df, H, lexm, [20], cfg, rng, device, lambda *a: None)
    res = pd.DataFrame(res)
    sc = res[(res.probe == "logreg") & (res.set == "swap_city")].bacc
    ck("подложенный сигнал по полю: на swap_city (метка по городу) проба ошибается", len(sc) > 0 and float(sc.iloc[0]) < 0.3,
       f"{float(sc.iloc[0]):.2f}" if len(sc) else "")
    sh = res[res.probe.str.startswith("shuf") & (res.set == "iid_test")].bacc
    ck.warn("подложенный сигнал: контроль с перемешанными метками около шанса", sh.isna().all() or float(sh.mean()) < 0.65)
    pos = res[(res.probe == "pos") & (res.set == "iid_test")].bacc
    ck.warn("подложенный сигнал: бейзлайн по позиции не читает метку", pos.isna().all() or float(pos.iloc[0]) < 0.7)


def check_outputs(ck, cfg, res, pr):
    """Файлы результата и разумность итогов на реальных состояниях."""
    files = ["rows.csv", "run.log"]
    if res is not None:
        files += ["probe_results.csv", "summary.csv", "dirs.csv"] + [f"{n}.png" for n in
                  ("01_controls", "02_role", "03_pointer", "04_pointer_errors", "05_used", "06_used_swap", "07_used_xslot",
                   "08_used_distance", "09_conflict", "10_saturation", "11_dirs")]
    if pr is not None:
        files += ["patch.csv", "patch_summary.csv"] + [f"{12 + i}_patch_{k}.png" for i, k in enumerate(pr.kind.unique())]
    for f in files:
        ck(f"файл {f} создан", os.path.exists(f"{cfg.OUT_DIR}/{f}"))
    if res is not None:
        nan = res[(res.probe == "logreg") & (res.set == "iid_test")].bacc.isna().sum()
        ck.warn("в iid_test нет наборов с одним классом (bacc = NaN)", nan == 0, f"{nan} ячеек (при малой выборке это ожидаемо)")
        q0 = res[(res.question == "Q0_marker") & (res.probe == "logreg") & (res.set == "iid_test") & (res.layer >= 0)]
        ck.warn("Q0_marker (проверка на баг): маркер роли читается на токене города, лучший слой ≥ 0.9", len(q0) > 0 and q0.bacc.max() >= 0.9,
                f"макс. {q0.bacc.max():.2f}" if len(q0) else "")
    if pr is not None and len(pr):
        for k, g in [("marker", "markers"), ("city", "cities")]:
            x = pr[(pr.kind == k) & (pr.group == g) & (pr.how == "full") & (pr.layer == 0)]
            ck.warn(f"патч {k}/{g} на слое 0 воспроизводит контрфакт (nld ≈ 1)", len(x) == 0 or abs(x.nld.mean() - 1) < 0.05,
                    f"nld {x.nld.mean():.2f}" if len(x) else "нет слоя 0")


def check_pairs(ck, lm, pairs):
    """Пары для патчинга: одинаковая длина, ответы различаются, токены различаются только там, где задумано."""
    ck("патчинг: пары собраны", len(pairs) > 0, f"{len(pairs)}")
    bad = {"marker": 0, "city": 0, "schema": 0}
    for pr in pairs:
        P, ids = pr["P"], pr["ids"]
        for kind, (ids_c, ans_c) in pr["cfs"].items():
            diff = {i for i, (a, b) in enumerate(zip(ids, ids_c)) if a != b}
            if kind == "marker":
                ok = diff == set(P["markers"])
            elif kind == "city":
                ok = diff and diff <= set(P["cities"])
            else:
                msg = min(P["markers"])
                ok = bool(P["key"]) and set(P["key"]) <= diff and not diff & set(P["cities"] + P["markers"] + P["slot"]) and \
                     all(i < msg or i in P["key"] for i in diff)
            bad[kind] += not (ok and len(ids_c) == len(ids) and ans_c != pr["ans"])
    for kind, n in bad.items():
        ck(f"патчинг, контрфакт {kind}: та же длина, другой ответ, токены отличаются только в задуманных позициях", n == 0, f"ошибок {n}")


def check_patch(ck, lm, pairs):
    """Патч на слое 0 в отличающихся позициях воспроизводит контрфакт; патч своими же состояниями ничего не меняет;
    формула патча вдоль направления меняет только проекцию на него."""
    if not pairs:
        ck("патчинг: есть пара для проверки хуков", False)
        return
    pr = pairs[0]
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
    allpos = sorted({p for v in pr["P"].values() for p in v})
    _, st_b = lm.run_states(lm.tensor(pr["ids"]), {p: p for p in allpos})
    base = lm.fwd_last(lm.tensor(pr["ids"]))[0]
    C = torch.stack([st_b[p] for p in pr["P"]["slot"]], 1)
    same = patching.patched_last(lm, pr["ids"], [0, lm.n_layers // 2], pr["P"]["slot"], C)
    ck("патч собственными состояниями не меняет логиты", rel(same[1], base) < 2e-2 and int(same[1].argmax()) == int(base.argmax()),
       f"отн. разница {rel(same[1], base):.1e}")
    for kind, grp in [("marker", "markers"), ("city", "cities")]:
        ids_c, _ = pr["cfs"][kind]
        lc, st = lm.run_states(lm.tensor(ids_c), {p: p for p in allpos})
        C = torch.stack([st[p] for p in pr["P"][grp]], 1)
        lp = patching.patched_last(lm, pr["ids"], [0], pr["P"][grp], C)[0]
        ck(f"патч {kind}: подмена группы {grp} на слое 0 воспроизводит контрфакт", rel(lp, lc[-1].float()) < 2e-2
           and int(lp.argmax()) == int(lc[-1].argmax()), f"отн. разница {rel(lp, lc[-1].float()):.1e}")
    g = torch.Generator().manual_seed(0)
    h, c, u = torch.randn(3, 16, generator=g), torch.randn(3, 16, generator=g), torch.randn(16, generator=g)
    u = u / u.norm()
    new = h + ((c - h) @ u)[:, None] * u
    ck("формула патча вдоль направления: проекция берётся у контрфакта, остальное у исходного",
       torch.allclose(new @ u, c @ u, atol=1e-5) and torch.allclose(new - (new @ u)[:, None] * u, h - (h @ u)[:, None] * u, atol=1e-5))
