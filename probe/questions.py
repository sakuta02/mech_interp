"""Таблица строк для проб (строка = пример × позиция) с метками и описание вопросов Q0..Q4: какие строки обучают,
на каких наборах оценивается проба и какая метка используется."""
import numpy as np, pandas as pd
from data import N_SLOTS, ROLES, FILLERS, NAMES

ROLE_ID = {"src": 0, "dst": 1, "other": 2}
ALL_KEYS = sorted({k for r in NAMES.values() for s in r.values() for k in s} | {f["key"] for f in FILLERS})
SETS_OOD = ["ood_city", "ood_name", "ood_desc", "ood_tpl", "ood_all"]


def build_rows(lm, specs, meta):
    """DataFrame строк: слоты 1..4 каждого примера и два токена городов; плюс матрица мультихота ключей (лексический бейзлайн)."""
    recs, lex = [], []
    for i, (sp, m) in enumerate(zip(specs, meta)):
        f = sp["fields"]
        roles = [x["role"] for x in f]
        rc = [r if r in ROLES else "other" for r in roles]
        tok = {r: lm.enc(sp[f"{r}_city"])[0] for r in ROLES}
        swap = sp["group"] == "swap"
        correct = swap or all(m["pred"][k] == lm.enc(f[k]["value"])[0] for k in range(N_SLOTS) if rc[k] != "other")
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
            recs.append(dict(base, site=k - 1, k=k, y_role=ROLE_ID[r], y_ord=k - 1, cur_city=r != "other", ptr_role=ptr,
                             ptr_pos=-1 if ptr < 0 else int(["src", "dst"][ptr] != sp["first"]),
                             gold_pos=-1 if r == "other" else int(r != sp["first"]), u_src=u["src"], u_dst=u["dst"],
                             uc_src=uc["src"], uc_dst=uc["dst"], pos_abs=m["pos"][k - 1]))
            lex.append(np.isin(ALL_KEYS, [x["key"] for x in f[:k]]))
        for j, r in enumerate(ROLES):
            recs.append(dict(base, site=N_SLOTS + j, k=0, y_role=ROLE_ID[r], y_ord=-1, cur_city=True, ptr_role=-1,
                             ptr_pos=-1, gold_pos=-1, u_src=-1, u_dst=-1, uc_src=-1, uc_dst=-1, pos_abs=m["cpos"][r]))
            lex.append(np.zeros(len(ALL_KEYS), bool))
    return pd.DataFrame(recs), np.array(lex, dtype=np.float32)


def make_questions(df, cfg):
    """Список вопросов: имя, строки задачи, колонка метки, число классов, fit/val и наборы оценки {имя: (маска, колонка метки)}.
    Q0_leak берёт только первый упомянутый город postfix-сообщения: у второго роль следует из маркера первого."""
    ok = df.correct if cfg.ONLY_CORRECT else pd.Series(True, index=df.index)
    iid, slot = df.group == "iid", df.site < N_SLOTS

    def q(name, base, y, ncls, extra=None, fit_extra=None, lex=True, evals=True):
        fe = fit_extra if fit_extra is not None else True
        ev = {"iid_test": (base & iid & (df.fit_split == "test") & ok, y)} if evals else {}
        if evals:
            for g in SETS_OOD:
                ev[g] = (base & (df.group == g) & ok, y)
            if cfg.ONLY_CORRECT:
                ev["iid_err"] = (base & iid & ~df.correct, y)
        ev.update(extra or {})
        return dict(name=name, base=base, y=y, ncls=ncls, lex=lex, evals=ev,
                    fit=base & iid & (df.fit_split == "fit") & ok & fe, val=base & iid & (df.fit_split == "val") & ok & fe)

    city = df.site >= N_SLOTS
    qs = [q("Q0_marker", city & (df.tpl_pos == "prefix"), "y_role", 2, lex=False),
          q("Q0_leak", city & (df.tpl_pos == "postfix") & (df.y_role == df.first_id), "y_role", 2, lex=False),
          q("Q1_role", slot, "y_role", 3),
          q("Q2_ordinal", slot, "y_ord", N_SLOTS)]
    for col, name in [("ptr_role", "Q3_ptr_role"), ("ptr_pos", "Q3_ptr_pos")]:
        b = slot & (df[col] >= 0)
        gold = "y_role" if col == "ptr_role" else "gold_pos"
        qs.append(q(name, b, col, 2, extra={"iid_err_gold": (b & iid & ~df.correct, gold)} if cfg.ONLY_CORRECT else None))
    for r in ROLES:
        b2 = slot & (df.k == 2)
        sw = b2 & (df.group == "swap")
        qs.append(q(f"Q4_used_{r}", b2, f"u_{r}", 2, extra={"swap_field": (sw, f"u_{r}"), "swap_city": (sw, f"uc_{r}")}))
        for nm, tr, te in [("o2c", ~df.cur_city, df.cur_city), ("c2o", df.cur_city, ~df.cur_city)]:
            t_ = b2 & iid & (df.fit_split == "test") & ok
            qs.append(q(f"Q4_used_{r}_{nm}", b2, f"u_{r}", 2, fit_extra=tr, evals=False,
                        extra={"xslot": (t_ & te, f"u_{r}"), "same_slot": (t_ & tr, f"u_{r}")}))
    return qs
