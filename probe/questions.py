"""Таблица строк для проб (строка = пример × позиция чтения) с метками и описание вопросов: какие строки обучают пробу,
на каких наборах она оценивается и какая метка используется на каждом наборе."""
import numpy as np, pandas as pd
from data import N_SLOTS, ROLES, OTHER, FILLERS, NAMES

ROLE_ID = {"src": 0, "dst": 1, "other": 2}
ALL_KEYS = sorted({k for r in NAMES.values() for s in r.values() for k in s} | {f["key"] for f in FILLERS})
SETS_OOD = ["ood_city", "ood_name", "ood_desc", "ood_tpl", "ood_all"]


def build_rows(lm, specs, meta):
    """DataFrame строк: слоты 1..4 каждого примера и два токена городов; плюс мультихот ключей (лексический бейзлайн)."""
    recs, lex = [], []
    for i, (sp, m) in enumerate(zip(specs, meta)):
        f = sp["fields"]
        roles = [x["role"] for x in f]
        rc = [r if r in ROLES else "other" for r in roles]
        tok = {r: lm.enc(sp[f"{r}_city"])[0] for r in ROLES}
        swap, conflict = sp["group"] == "swap", sp["group"] == "conflict"
        correct = swap or conflict or all(m["pred"][k] == lm.enc(f[k]["value"])[0] for k in range(N_SLOTS) if rc[k] != "other")
        base = dict(ex=i, group=sp["group"], fit_split=sp["fit_split"], tpl_pos=sp["tpl_pos"], correct=bool(correct),
                    first_id=ROLE_ID[sp["first"]])
        for k in range(1, N_SLOTS + 1):
            r = rc[k - 1]
            p = m["pred"][k - 1]
            ptr = -1 if (r == "other" or swap) else 0 if p == tok["src"] else 1 if p == tok["dst"] else -1
            u = {x: int(roles.index(x) + 1 < k) for x in ROLES}
            uc = dict(u)
            if swap and k == 2:
                uc = {x: int(x != roles[0]) for x in ROLES}
            dr = f[k - 1].get("desc_role", r) if r != "other" else "other"
            recs.append(dict(base, site=k - 1, k=k, y_role=ROLE_ID[r], desc_role=ROLE_ID[dr], y_ord=k - 1, cur_city=r != "other",
                             ptr_role=ptr, ptr_pos=-1 if ptr < 0 else int(["src", "dst"][ptr] != sp["first"]),
                             gold_pos=-1 if r == "other" else int(r != sp["first"]),
                             err_slot=bool(not swap and not conflict and ptr >= 0 and ptr != ROLE_ID[r]),
                             u_src=u["src"], u_dst=u["dst"], uc_src=uc["src"], uc_dst=uc["dst"],
                             dist_src=k - roles.index("src") - 1 if u["src"] else 0,
                             dist_dst=k - roles.index("dst") - 1 if u["dst"] else 0, pos_abs=m["pos"][k - 1]))
            lex.append(np.isin(ALL_KEYS, [x["key"] for x in f[:k]]))
        for j, r in enumerate(ROLES):
            recs.append(dict(base, site=N_SLOTS + j, k=0, y_role=ROLE_ID[r], desc_role=ROLE_ID[r], y_ord=-1, cur_city=True,
                             ptr_role=-1, ptr_pos=-1, gold_pos=-1, err_slot=False, u_src=-1, u_dst=-1, uc_src=-1, uc_dst=-1,
                             dist_src=0, dist_dst=0, pos_abs=m["cpos"][r]))
            lex.append(np.zeros(len(ALL_KEYS), bool))
    return pd.DataFrame(recs), np.array(lex, dtype=np.float32)


def make_questions(df, cfg):
    """Список вопросов: имя, строки задачи (base), колонка метки, число классов, fit/val и наборы оценки {имя: (маска, колонка)}.
    Строки base, не попавшие в fit/val, используются только в наборах оценки."""
    ok = df.correct if cfg.ONLY_CORRECT else pd.Series(True, index=df.index)
    slot, city = df.site < N_SLOTS, df.site >= N_SLOTS
    normal = ~df.group.isin(["swap", "conflict"])
    conf = (df.group == "conflict") & df.cur_city

    def part(g, s):
        """Строки группы g с fit_split = s."""
        return (df.group == g) & (df.fit_split == s)

    def q(name, base, y, ncls, grp="iid", extra=None, fit_extra=True, std=True, lex=True):
        """Один вопрос; std — добавить стандартные наборы iid_test и ood_*."""
        ev = {}
        if std:
            ev["iid_test"] = (base & part(grp, "test") & ok, y)
            for g in SETS_OOD:
                ev[g] = (base & (df.group == g) & ok, y)
        ev.update(extra or {})
        return dict(name=name, base=base, y=y, ncls=ncls, lex=lex, evals=ev,
                    fit=base & part(grp, "fit") & ok & fit_extra, val=base & part(grp, "val") & ok & fit_extra)

    # Q0_leak: только первый упомянутый город postfix-сообщения (у второго роль следует из маркера первого)
    qs = [q("Q0_marker", city & (df.tpl_pos == "prefix") & normal, "y_role", 2, lex=False),
          q("Q0_leak", city & (df.tpl_pos == "postfix") & (df.y_role == df.first_id) & normal, "y_role", 2, grp="postfix", lex=False)]
    b = slot & (df.group != "swap")
    qs += [q("Q1_role", b, "y_role", 3, extra={"err_slot": (b & df.err_slot, "y_role"),
                                              "conflict_key": (b & conf, "y_role"), "conflict_desc": (b & conf, "desc_role")}),
           q("Q2_ordinal", slot & normal, "y_ord", N_SLOTS)]
    for col, gold, name in [("ptr_role", "y_role", "Q3_ptr_role"), ("ptr_pos", "gold_pos", "Q3_ptr_pos")]:
        b = slot & (df[col] >= 0) & (df.group != "swap")
        extra = {"err_model": (b & df.err_slot, col), "err_gold": (b & df.err_slot, gold)}
        if col == "ptr_role":
            extra.update({"conflict_key": (b & conf, "y_role"), "conflict_desc": (b & conf, "desc_role")})
        qs.append(q(name, b, col, 2, extra=extra))
    t_iid = part("iid", "test") & ok
    for r in ROLES:
        u = f"u_{r}"
        b2 = slot & (df.k == 2) & (df.group != "conflict")
        sw = b2 & (df.group == "swap")
        qs.append(q(f"Q4_used_{r}", b2, u, 2, extra={"swap_field": (sw, u), "swap_city": (sw, f"uc_{r}")}))
        # перенос между видами слота; строки, где текущий слот сам роли r, исключены: там метка всегда 0
        bx = slot & (df.k == 2) & normal & (df.y_role != ROLE_ID[r])
        oth, cty = df.y_role == ROLE_ID["other"], df.y_role == ROLE_ID[OTHER[r]]
        for nm, tr, te in [("o2c", oth, cty), ("c2o", cty, oth)]:
            qs.append(q(f"Q4_used_{r}_{nm}", bx, u, 2, fit_extra=tr, std=False,
                        extra={"xslot": (bx & t_iid & te, u), "same_slot": (bx & t_iid & tr, u)}))
        # «использован» на слотах 2..4: dist{d} — все отрицательные плюс положительные, где поле роли r было d полей назад
        bd = slot & df.k.isin([2, 3, 4]) & normal
        qs.append(q(f"Q4d_used_{r}", bd, u, 2,
                    extra={f"dist{d}": (bd & t_iid & ((df[u] == 0) | (df[f"dist_{r}"] == d)), u) for d in (1, 2, 3)}))
    return qs
