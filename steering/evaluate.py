"""Основной прогон: для каждого тестового примера считает базовый ответ модели, ответ при добавке каждого вектора на каждом слое
и при подмене активации, и сохраняет всё в raw.csv.
"""
import itertools, random, time
import pandas as pd, torch

from data import TOOLS, TEMPLATES
from model import RESID, steer
from prompts import prefix, arg_seq


def eval_example(lm, cfg, vecs_by_tool, t, tpl, O, D, X, slot):
    """Один пример: базовый ответ, добавка каждого вектора на каждом слое и подмена активации; возвращает строки для raw.csv.
    S — своё значение слота, T — целевое (другой роли), X — третье.
    """
    ids, role, tok1 = prefix(lm, t, tpl, O, D, X, slot)
    own, oth = (O, D) if role == "o" else (D, O)
    s_id, t_id, x_id = lm.enc(own)[0], lm.enc(oth)[0], lm.enc(X)[0]
    sgn = 1.0 if role == "o" else -1.0
    x1 = lm.tensor(ids)
    base = lm.fwd_last(x1)[0].cpu()
    ids_sw, _, _ = prefix(lm, t, tpl, D, O, X, slot)
    sw = lm.run_states(lm.tensor(ids_sw), {"p": len(ids_sw) - 1}, logits=False)[1]["p"]
    r_first = [r for r, _ in arg_seq(t) if r != "x"][0]
    first_ok = None if tok1 is None else tok1 == lm.enc(O if r_first == "o" else D)[0]
    meta = dict(tool_id=t["id"], pair=t["name"], domain=t["domain"], order=t["order"], layout=t["layout"],
                tool_split=t["split"], slot=slot, role=role, tpl=tpl["id"], tpl_type=tpl["type"],
                tpl_split=tpl["split"], o_first=tpl["o_first"], O=O, D=D, X=X,
                base_ok=base.argmax().item() == s_id, first_ok=first_ok,
                base_delta=(base[t_id] - base[s_id]).item())

    def rec(lg, layer, cond, alpha):
        """Строка результата: сдвиг логитов и что стало топ-1 (T, S или X)."""
        top = lg.argmax().item()
        return dict(meta, layer=layer, cond=cond, alpha=alpha,
                    delta=(lg[t_id] - lg[s_id]).item(), delta_tx=(lg[t_id] - lg[x_id]).item(),
                    top1_is_T=top == t_id, top1_is_S=top == s_id, top1_is_X=top == x_id)

    rows = [rec(base, -1, "none", 0.0)]
    spec = [(c, a, v) for c, v, al in vecs_by_tool[t["id"]] for a in al]
    for l in cfg.LAYER_RANGE:
        for k in range(0, len(spec), cfg.BATCH):
            chunk = spec[k:k + cfg.BATCH]
            adds = torch.stack([sgn * a * v[l] for _, a, v in chunk])
            lg = lm.fwd_last(x1.repeat(len(chunk), 1), [(RESID(l), steer(adds=adds))]).cpu()
            rows += [rec(lg[b], l, c, a) for b, (c, a, _) in enumerate(chunk)]
        lg = lm.fwd_last(x1, [(RESID(l), steer(repl=sw[l]))]).cpu()
        rows.append(rec(lg[0], l, "patch", float("nan")))
    return rows


def run(lm, cfg, vecs_by_tool, pool, dom, n, done, log):
    """Прогоняет n примеров одного типа задач; done — строки уже пройденных типов для partial.csv."""
    tools_ = [t for t in TOOLS if t["domain"] == dom]
    tpls_ = [x for x in TEMPLATES if x["domain"] == dom]
    combos = list(itertools.product(tools_, tpls_, ["first", "second"])); random.shuffle(combos)
    out, t0 = [], time.time()
    log(f"эксперимент {dom}: {n} примеров")
    for i in range(n):
        t, tpl, slot = combos[i % len(combos)]
        O, D, X = random.sample(pool[dom]["test"], 3)
        out += eval_example(lm, cfg, vecs_by_tool, t, tpl, O, D, X, slot)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            log(f"{dom}: {i + 1}/{n}, на пример {el / (i + 1):.1f} с, осталось {el / (i + 1) * (n - i - 1) / 60:.1f} мин")
        if (i + 1) % 20 == 0:
            pd.DataFrame(done + out).to_csv(f"{cfg.OUT_DIR}/partial.csv", index=False)
    return out


def run_all(lm, cfg, vecs_by_tool, pool, log):
    """Прогоняет все типы задач и пишет raw.csv."""
    rows = []
    for dom, n in [("flight", cfg.N_TEST), ("message", cfg.N_TEST_DOM), ("period", cfg.N_TEST_DOM)]:
        rows += run(lm, cfg, vecs_by_tool, pool, dom, n, rows, log)
    df = pd.DataFrame(rows)
    df.to_csv(f"{cfg.OUT_DIR}/raw.csv", index=False)
    log("эксперимент завершён, raw.csv сохранён, строк:", len(rows))
    return df
