"""Основной прогон опыта: для каждого тестового примера базовый ответ модели, ответ при добавке каждого вектора
на каждом слое и при подмене активации; всё сохраняется в raw.csv."""
import itertools, random, time
import pandas as pd, torch

from data import N_ARGS, TOOLS, TEMPLATES, TRANSITIONS, sample_vals
from model import RESID, steer
from prompts import call_ids


def eval_example(lm, cfg, vecs, t, tpl, tr, vals):
    """Один пример перехода tr = (o, d): контекст до значения аргумента o, цель T — значение аргумента d, своё S — аргумента o,
    R — значения остальных трёх аргументов. Возвращает строки для raw.csv."""
    o, d = tr
    ids, pos = call_ids(lm, t, tpl, vals, upto=o)
    x1 = lm.tensor(ids)
    s_id, t_id = lm.enc(vals[o])[0], lm.enc(vals[d])[0]
    r_ids = [lm.enc(vals[k])[0] for k in range(1, N_ARGS + 1) if k not in tr]
    ctx_ok = True
    if o > 1:
        logits_all = lm.model(x1)[0]
        ctx_ok = all(logits_all[pos[k]].argmax().item() == lm.enc(vals[k])[0] for k in range(1, o))
    base = lm.fwd_last(x1)[0].cpu()
    vals_sw = dict(vals); vals_sw[o], vals_sw[d] = vals[d], vals[o]
    ids_sw, _ = call_ids(lm, t, tpl, vals_sw, upto=o)
    sw = lm.run_states(lm.tensor(ids_sw), {"p": len(ids_sw) - 1}, logits=False)[1]["p"]
    meta = dict(tool_id=t["id"], trans=f"{o}_{d}", o=o, d=d, type_o=t["args"][o - 1]["type"], type_d=t["args"][d - 1]["type"],
                tpl=tpl["id"], tpl_style=tpl["style"], tpl_rev=tpl["rev"], tpl_split=tpl["split"],
                ex=" ".join(vals[k] for k in range(1, N_ARGS + 1)), S=vals[o], T=vals[d],
                ctx_ok=ctx_ok, base_ok=base.argmax().item() == s_id, base_delta=(base[t_id] - base[s_id]).item())

    def rec(lg, layer, cond, kind, alpha):
        """Строка результата: сдвиг логитов и что стало топ-1 (T, S или R)."""
        top = lg.argmax().item()
        return dict(meta, layer=layer, cond=cond, kind=kind, alpha=alpha, delta=(lg[t_id] - lg[s_id]).item(),
                    top1_is_T=top == t_id, top1_is_S=top == s_id, top1_is_R=top in r_ids)

    rows = [rec(base, -1, "none", "none", 0.0)]
    spec = [(c, k, a, v) for c, k, v, al in vecs[(t["id"], tr)] for a in al]
    for l in cfg.LAYER_RANGE:
        for j in range(0, len(spec), cfg.BATCH):
            chunk = spec[j:j + cfg.BATCH]
            adds = torch.stack([a * v[l] for _, _, a, v in chunk])
            lg = lm.fwd_last(x1.repeat(len(chunk), 1), [(RESID(l), steer(adds=adds))]).cpu()
            rows += [rec(lg[b], l, c, k, a) for b, (c, k, a, _) in enumerate(chunk)]
        lg = lm.fwd_last(x1, [(RESID(l), steer(repl=sw[l]))]).cpu()
        rows.append(rec(lg[0], l, "patch", "patch", float("nan")))
    return rows


def run_all(lm, cfg, vecs, pool, log):
    """N_TEST примеров по кругу из перемешанных комбинаций (инструмент, шаблон, переход); пишет raw.csv."""
    combos = list(itertools.product(TOOLS, TEMPLATES, TRANSITIONS)); random.shuffle(combos)
    rows, t0 = [], time.time()
    log(f"эксперимент: {cfg.N_TEST} примеров, комбинаций {len(combos)}")
    for i in range(cfg.N_TEST):
        t, tpl, tr = combos[i % len(combos)]
        rows += eval_example(lm, cfg, vecs, t, tpl, tr, sample_vals(t, pool, "test"))
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            log(f"{i + 1}/{cfg.N_TEST}, на пример {el / (i + 1):.1f} с, осталось {el / (i + 1) * (cfg.N_TEST - i - 1) / 60:.1f} мин")
        if (i + 1) % 20 == 0:
            pd.DataFrame(rows).to_csv(f"{cfg.OUT_DIR}/partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(f"{cfg.OUT_DIR}/raw.csv", index=False)
    log("эксперимент завершён, raw.csv сохранён, строк:", len(rows))
    return df
