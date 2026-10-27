"""Построение векторов опыта: калибровка состояний слотов 1, 3, 5, вектор перехода o -> d для каждого инструмента,
вектор тех же переходов по остальным инструментам («неродной»), контрольные векторы и их геометрия."""
import json, random
from dataclasses import dataclass
import pandas as pd, torch

from data import TOOLS, TEMPLATES, TRANSITIONS, SLOTS, sample_vals
from prompts import call_ids


@dataclass
class Vectors:
    """Векторы переходов, формы [L, d]: native — свой вектор перехода (инструмент, (o, d));
    foreign — тот же переход по всем остальным инструментам; shuf — контроль с обменом меток, без проекции на native."""
    native: dict
    foreign: dict
    shuf: dict


def calibrate(lm, cfg, t, n, pool):
    """Состояния слотов 1, 3 и 5 инструмента при teacher forcing: {k: [kept, L, d]}.
    При FILTER_CALIB остаются примеры, где модель верна во всех трёх слотах."""
    tpls = [x for x in TEMPLATES if x["split"] == "calib"]
    out = {k: [] for k in SLOTS}
    for i in range(n):
        vals = sample_vals(t, pool, "calib")
        ids, pos = call_ids(lm, t, tpls[i % len(tpls)], vals)
        logits, st = lm.run_states(lm.tensor(ids), {k: pos[k] for k in SLOTS})
        if cfg.FILTER_CALIB and any(logits[pos[k]].argmax().item() != lm.enc(vals[k])[0] for k in SLOTS):
            continue
        for k in SLOTS:
            out[k].append(st[k])
    assert out[SLOTS[0]], f"{t['id']}: все калибровочные примеры отфильтрованы"
    return {k: torch.stack(v) for k, v in out.items()}


def collect(lm, cfg, pool, log):
    """Калибровка всех инструментов: {id инструмента: {слот: [kept, L, d]}}."""
    H = {}
    for i, t in enumerate(TOOLS):
        H[t["id"]] = calibrate(lm, cfg, t, cfg.N_PER_TOOL, pool)
        log(f"калибровка {i + 1}/{len(TOOLS)}: {t['id']}, осталось примеров {H[t['id']][SLOTS[0]].shape[0]}/{cfg.N_PER_TOOL}")
    return H


def pooled(H, tool_ids, o, d):
    """Вектор перехода o -> d по объединённым примерам инструментов: среднее состояние слота d минус слота o."""
    return torch.cat([H[i][d] for i in tool_ids]).mean(0) - torch.cat([H[i][o] for i in tool_ids]).mean(0)


def unit(v, ref):
    """Направление v с нормой ref на каждом слое."""
    return v / v.norm(dim=-1, keepdim=True) * ref.norm(dim=-1, keepdim=True)


def rand_unit(seed, ref):
    """Случайное направление формы ref с нормой ref по seed."""
    g = torch.Generator(device=ref.device).manual_seed(seed)
    return unit(torch.randn(ref.shape, generator=g, device=ref.device), ref)


def build_vectors(H):
    """Свои, неродные и перемешанные векторы для каждой пары (инструмент, переход)."""
    ids = list(H)
    native = {(i, tr): pooled(H, [i], *tr) for i in ids for tr in TRANSITIONS}
    foreign = {(i, tr): pooled(H, [j for j in ids if j != i], *tr) for i in ids for tr in TRANSITIONS}
    shuf = {}
    for i in ids:
        n = H[i][SLOTS[0]].shape[0]
        mask = torch.zeros(n, dtype=torch.bool, device=H[i][SLOTS[0]].device)
        mask[torch.randperm(n, device=mask.device)[: n // 2]] = True
        m = mask[:, None, None]
        for tr in TRANSITIONS:
            Ho, Hd = H[i][tr[0]], H[i][tr[1]]
            ds = torch.where(m, Ho, Hd).mean(0) - torch.where(m, Hd, Ho).mean(0)
            c = native[(i, tr)] / native[(i, tr)].norm(dim=-1, keepdim=True)
            shuf[(i, tr)] = ds - (ds * c).sum(-1, keepdim=True) * c
    return Vectors(native=native, foreign=foreign, shuf=shuf)


def build_vecs(cfg, V):
    """Для каждой пары (инструмент, переход) список (условие, вид, вектор, силы); все векторы с нормой своего native."""
    out = {}
    for t in TOOLS:
        for tr in TRANSITIONS:
            o, d = tr
            ref = V.native[(t["id"], tr)]
            out[(t["id"], tr)] = [
                (f"steer_tool_{o}_{d}",    "steer_tool",    ref,                                 cfg.ALPHAS),        # свой вектор перехода
                (f"steer_foreign_{o}_{d}", "steer_foreign", unit(V.foreign[(t["id"], tr)], ref), cfg.ALPHAS),        # тот же переход по остальным инструментам
                ("steer_neg", "steer_neg", -ref,                                                 cfg.CTRL_ALPHAS),   # контроли
                ("shuffled",  "shuffled",  unit(V.shuf[(t["id"], tr)], ref),                     cfg.CTRL_ALPHAS),
                ("random0",   "random",    rand_unit(0, ref),                                   cfg.CTRL_ALPHAS),
                ("random1",   "random",    rand_unit(1, ref),                                   cfg.CTRL_ALPHAS)]
    return out


def geometry(cfg, V):
    """По слоям LAYER_RANGE: норма вектора, согласие векторов разных инструментов и косинус native с foreign."""
    cos = lambda a, b: torch.cosine_similarity(a, b, dim=0).item()
    ids, K = [t["id"] for t in TOOLS], len(TOOLS)
    fwd = [tr for tr in TRANSITIONS if tr[0] < tr[1]]
    geom = []
    for l in cfg.LAYER_RANGE:
        r = dict(layer=l)
        for tr in fwd:
            k = f"{tr[0]}>{tr[1]}"
            vs = torch.stack([V.native[(i, tr)][l] for i in ids])
            vn = vs / vs.norm(dim=-1, keepdim=True)
            r[f"norm[{k}]"] = vs.norm(dim=-1).mean().item()
            r[f"cos_tools[{k}]"] = ((vn @ vn.T).sum().item() - K) / (K * (K - 1))
            r[f"cos_foreign[{k}]"] = sum(cos(V.native[(i, tr)][l], V.foreign[(i, tr)][l]) for i in ids) / K
        for a, b in zip(fwd, fwd[1:]):
            r[f"cos_trans[{a[0]}>{a[1]}~{b[0]}>{b[1]}]"] = sum(cos(V.native[(i, a)][l], V.native[(i, b)][l]) for i in ids) / K
        geom.append(r)
    return pd.DataFrame(geom).round(3).set_index("layer")


def save_artifacts(cfg, V, H, pool, geom, log):
    """Сохраняет геометрию, векторы и настройки прогона."""
    out = cfg.OUT_DIR
    geom.to_csv(f"{out}/vector_geometry.csv")
    print(geom.T.to_string())
    pack = lambda d: {f"{k[0]}|{k[1][0]}>{k[1][1]}": v.cpu() for k, v in d.items()}
    torch.save({"native": pack(V.native), "foreign": pack(V.foreign), "shuf": pack(V.shuf)}, f"{out}/vectors.pt")
    json.dump(dict(vars(cfg), POOL=pool, kept_calib={k: int(v[SLOTS[0]].shape[0]) for k, v in H.items()}),
              open(f"{out}/config.json", "w"), indent=1,
              default=lambda r: [r.start, r.stop] if isinstance(r, range) else str(r))
    log("векторы построены, геометрия и векторы сохранены")
