"""Построение векторов: калибровка состояний слотов, вектор роли слота, контрольные векторы и их геометрия."""
import json, random
from dataclasses import dataclass
import pandas as pd, torch

from data import TOOLS, TEMPLATES, LAYOUTS
from prompts import tf_ids

SIGMA = {"od": 1.0, "do": -1.0}


@dataclass
class Vectors:
    """Построенные векторы, формы [L, d] (norm_s — [L, 1]): s — роль без позиции; D_cfg — по (порядок ключей, вариант вызова);
    e_lay — часть, зависящая от порядка; dom_vec — по типу задач; d_shuf — контроль с обменом меток; per_tool — по инструменту.
    """
    s: torch.Tensor
    norm_s: torch.Tensor
    D_cfg: dict
    e_lay: dict
    dom_vec: dict
    d_shuf: torch.Tensor
    per_tool: dict


def calibrate(lm, cfg, t, n, pool):
    """Состояния слотов origin и destination инструмента при teacher forcing: ([kept, L, d], [kept, L, d]).
    При FILTER_CALIB остаются примеры, где модель верна в обоих слотах.
    """
    tpls = [x for x in TEMPLATES if x["domain"] == t["domain"] and x["split"] == "calib"]
    pool = pool[t["domain"]]["calib"]
    Ho, Hd = [], []
    for i in range(n):
        O, D, X = random.sample(pool, 3)
        ids, pos = tf_ids(lm, t, tpls[i % len(tpls)], O, D, X)
        logits, st = lm.run_states(ids, pos)
        tgt = {"o": lm.enc(O)[0], "d": lm.enc(D)[0]}
        if cfg.FILTER_CALIB and any(logits[pos[r]].argmax().item() != tgt[r] for r in "od"):
            continue
        Ho.append(st["o"]); Hd.append(st["d"])
    assert Ho, f"{t['id']}: все калибровочные примеры отфильтрованы"
    return torch.stack(Ho), torch.stack(Hd)


def collect(lm, cfg, pool, log):
    """Калибровка всех инструментов."""
    H = {}
    for i, t in enumerate(TOOLS):
        H[t["id"]] = calibrate(lm, cfg, t, cfg.N_PER_TOOL, pool)
        log(f"калибровка {i + 1}/{len(TOOLS)}: {t['id']}, осталось примеров {H[t['id']][0].shape[0]}/{cfg.N_PER_TOOL}")
    return H


role_vec = lambda Ho, Hd: Hd.mean(0) - Ho.mean(0)


def unit(v, norm_s):
    """Вектор v с нормой ||s|| на каждом слое."""
    return v / v.norm(dim=-1, keepdim=True) * norm_s


def rand_unit(seed, s, norm_s):
    """Случайный вектор формы s и нормы ||s|| по seed."""
    g = torch.Generator(device=s.device).manual_seed(seed)
    return unit(torch.randn(s.shape, generator=g, device=s.device), norm_s)


def build_vectors(H):
    """Все векторы по калибровочным состояниям H."""
    def pooled(ids_):
        """Вектор роли по набору инструментов: среднее состояние второго слота минус первого."""
        return role_vec(torch.cat([H[i][0] for i in ids_]), torch.cat([H[i][1] for i in ids_]))

    FL = [t for t in TOOLS if t["domain"] == "flight"]
    D_cfg = {(o, lay): pooled([t["id"] for t in FL if t["split"] == "calib" and t["order"] == o and t["layout"] == lay])
             for o in ("od", "do") for lay in LAYOUTS}
    s = torch.stack(list(D_cfg.values())).mean(0)
    e_lay = {lay: (D_cfg[("od", lay)] - D_cfg[("do", lay)]) / 2 for lay in LAYOUTS}
    dom_vec = {dom: pooled([t["id"] for t in TOOLS if t["domain"] == dom]) for dom in ("message", "period")}
    norm_s = s.norm(dim=-1, keepdim=True)

    cal_ids = [t["id"] for t in FL if t["split"] == "calib"]
    Ho_all = torch.cat([H[i][0] for i in cal_ids]); Hd_all = torch.cat([H[i][1] for i in cal_ids])
    n_all = Ho_all.shape[0]
    mask = torch.zeros(n_all, dtype=torch.bool, device=Ho_all.device)
    mask[torch.randperm(n_all, device=Ho_all.device)[: n_all // 2]] = True
    m = mask[:, None, None]
    d_shuf = torch.where(m, Ho_all, Hd_all).mean(0) - torch.where(m, Hd_all, Ho_all).mean(0)
    c = s / norm_s
    d_shuf = d_shuf - (d_shuf * c).sum(-1, keepdim=True) * c

    return Vectors(s=s, norm_s=norm_s, D_cfg=D_cfg, e_lay=e_lay, dom_vec=dom_vec, d_shuf=d_shuf,
                   per_tool={k: role_vec(*v) for k, v in H.items()})


def build_vecs_by_tool(cfg, V):
    """Для каждого инструмента список (условие, вектор, силы)."""
    s, ns = V.s, V.norm_s
    common_ctrl = [("steer_neg", -unit(s, ns), cfg.CTRL_ALPHAS), ("shuffled", unit(V.d_shuf, ns), cfg.CTRL_ALPHAS),
                   ("random0", rand_unit(0, s, ns), cfg.CTRL_ALPHAS), ("random1", rand_unit(1, s, ns), cfg.CTRL_ALPHAS)]
    out = {}
    for t in TOOLS:
        o, lay = t["order"], t["layout"]
        opp = "do" if o == "od" else "od"
        v = [("steer",      unit(s, ns),                          cfg.ALPHAS),   # роль без позиции (вектор рейсов)
             ("steer_same", unit(V.D_cfg[(o, lay)], ns),          cfg.ALPHAS),   # тот же порядок и вариант вызова
             ("steer_opp",  unit(V.D_cfg[(opp, lay)], ns),        cfg.ALPHAS),   # противоположный порядок
             ("steer_tool", unit(V.per_tool[t["id"]], ns),        cfg.ALPHAS)]   # вектор самого инструмента
        if t["domain"] == "flight":
            others = [l for l in LAYOUTS if l != lay]
            v += [("steer_layout", unit(torch.stack([V.D_cfg[(o, l)] for l in others]).mean(0), ns), cfg.ALPHAS),  # другие варианты вызова
                  ("steer_pos",    unit(SIGMA[o] * V.e_lay[lay], ns), cfg.ALPHAS)]                                # позиция без роли
        else:
            v += [("steer_dom", unit(V.dom_vec[t["domain"]], ns), cfg.ALPHAS)]      # вектор самого типа задач (контроль)
        out[t["id"]] = v + common_ctrl
    return out


def geometry(cfg, V):
    """Нормы и косинусы между векторами по слоям LAYER_RANGE."""
    cos = lambda a, b: torch.cosine_similarity(a, b, dim=0).item()
    geom = []
    for l in cfg.LAYER_RANGE:
        r = dict(layer=l, norm_s=V.s[l].norm().item(), cos_shuf_s=cos(V.d_shuf[l], V.s[l]))
        for lay in LAYOUTS:
            r[f"cos_od_do[{lay}]"] = cos(V.D_cfg[("od", lay)][l], V.D_cfg[("do", lay)][l])
            r[f"e_over_s[{lay}]"] = V.e_lay[lay][l].norm().item() / V.s[l].norm().item()
        for o in ("od", "do"):
            for a, b in [("none", "date_mid"), ("none", "date_first"), ("date_mid", "date_first")]:
                r[f"cos_{o}[{a}~{b}]"] = cos(V.D_cfg[(o, a)][l], V.D_cfg[(o, b)][l])
        for dom in V.dom_vec:
            r[f"cos_dom[{dom}~s]"] = cos(V.dom_vec[dom][l], V.s[l])
        geom.append(r)
    return pd.DataFrame(geom).round(3).set_index("layer")


def save_artifacts(cfg, V, H, pool, geom, log):
    """Сохраняет геометрию, векторы и настройки прогона."""
    out = cfg.OUT_DIR
    geom.to_csv(f"{out}/vector_geometry.csv")
    print(geom.T.to_string())
    torch.save({"D_cfg": {f"{k[0]}|{k[1]}": v.cpu() for k, v in V.D_cfg.items()}, "s": V.s.cpu(),
                "e_lay": {k: v.cpu() for k, v in V.e_lay.items()}, "dom_vec": {k: v.cpu() for k, v in V.dom_vec.items()},
                "d_shuf": V.d_shuf.cpu(), "per_tool": {k: v.cpu() for k, v in V.per_tool.items()}},
               f"{out}/vectors.pt")
    json.dump(dict(vars(cfg), POOL=pool, kept_calib={k: int(v[0].shape[0]) for k, v in H.items()}),
              open(f"{out}/config.json", "w"), indent=1,
              default=lambda r: [r.start, r.stop] if isinstance(r, range) else str(r))
    log("векторы построены, геометрия и векторы сохранены")
