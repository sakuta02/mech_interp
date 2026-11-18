"""Патчинг: в исходном прогоне resid_pre слоя l в группе позиций подменяется активацией контрфакта той же длины
(целиком, только вдоль направления пробы или вдоль случайного направления). Смотрим, переворачивается ли ответ
в первом городском слоте: доля переворота и нормированная разность логитов."""
import random
import pandas as pd, torch
from data import CF_KINDS, ROLES, counterfactual, make_spec
from prompts import call_ids

# что патчим для каждого контрфакта: (группа позиций, как). full — вся активация; dir:<вопрос> — только компонента вдоль
# направления пробы этого вопроса; rand:<вопрос> — вдоль случайного направления (контроль)
RUNS = {
    "marker": [("cities", "full"), ("cities", "dir:Q0_marker"), ("cities", "rand:Q0_marker"), ("markers", "full"),
               ("slot", "full"), ("slot", "dir:Q3_ptr_pos"), ("slot", "rand:Q3_ptr_pos")],
    "city": [("cities", "full"), ("slot", "full"), ("slot", "dir:Q3_ptr_pos"), ("slot", "rand:Q3_ptr_pos")],
    "schema": [("key", "full"), ("slot", "full"), ("slot", "dir:Q3_ptr_role"), ("slot", "rand:Q3_ptr_role")],
}
DIR_QUESTIONS = ("Q0_marker", "Q3_ptr_pos", "Q3_ptr_role")


def positions(lm, sp):
    """Токены до первого городского слота включительно, группы позиций и ответ (первый токен верного города в этом слоте).
    Группы (позиции по возрастанию): slot — сам слот; cities — все токены обоих городов в сообщении; markers — токен
    перед каждым городом (from / to).
    Группа key (токены названия поля в вызове) добавляется в make_pairs по паре schema."""
    k = next(j for j, f in enumerate(sp["fields"], 1) if f["role"] in ROLES)
    ids, pos, cpos = call_ids(lm, sp, upto=k)
    cities = sorted(i for r in ROLES for i in range(cpos[r + "_first"], cpos[r] + 1))
    P = dict(slot=[pos[k]], cities=cities, markers=sorted(cpos[r + "_first"] - 1 for r in ROLES))
    return ids, P, lm.enc(sp["fields"][k - 1]["written"])[0]


def make_pairs(lm, cfg, pools, log):
    """Примеры iid с шаблонами from_to и to_from, у которых все три контрфакта той же длины, с теми же позициями групп
    и с другим ответом в первом городском слоте. key — позиции вызова (после сообщения), где пара schema отличается:
    это токены названия поля, у которых общий хвост (например, _city) не входит в группу."""
    rng, pairs, tries = random.Random(cfg.SEED + 7), [], 0
    while len(pairs) < cfg.N_PATCH and tries < 50 * cfg.N_PATCH:
        tries += 1
        sp = make_spec(rng, pools, "iid", tpl_ids=("from_to", "to_from"))
        ids, P, ans = positions(lm, sp)
        cfs = {}
        for kind in CF_KINDS:
            ids_c, P_c, ans_c = positions(lm, counterfactual(sp, kind))
            if len(ids_c) != len(ids) or P_c != P or ans_c == ans:
                break
            cfs[kind] = (ids_c, ans_c)
        else:
            ids_s = cfs["schema"][0]
            P["key"] = [i for i, (a, b) in enumerate(zip(ids, ids_s)) if a != b and i > max(P["cities"])]
            if P["key"]:
                pairs.append(dict(spec=sp, ids=ids, P=P, ans=ans, cfs=cfs))
    log(f"патчинг: {len(pairs)} примеров за {tries} попыток")
    return pairs


def patched_last(lm, ids, layers, posl, C, u=None):
    """Батч из len(layers) копий исходного прогона: в строке b на слое layers[b] в позициях posl resid_pre заменяется
    на C[l] целиком (u = None) или только компонента вдоль u[l]: h ← h + ((c − h)·u)u. Логиты последней позиции [B, V]."""
    row = {l: b for b, l in enumerate(layers)}

    def patch(act, hook):
        """Подменяет активацию в строке своего слоя."""
        l = hook.layer()
        if l in row:
            b = row[l]
            h, c = act[b, posl].float(), C[l]
            new = c if u is None else h + ((c - h) @ u[l])[:, None] * u[l]
            act[b, posl] = new.to(act.dtype)
        return act
    return lm.fwd_last(lm.tensor(ids).repeat(len(layers), 1), [(lambda n: n.endswith("hook_resid_pre"), patch)])


def run_patching(lm, cfg, pairs, dirs, log):
    """Все пары, контрфакты и варианты патча по слоям PATCH_LAYERS. В анализ идут пары, где модель верна и в исходном
    прогоне, и в контрфакте. Метрики строки: flip (argmax = ответ контрфакта) и nld = (LD_p − LD_b) / (LD_c − LD_b),
    LD = z[ответ контрфакта] − z[исходный ответ]."""
    layers = list(cfg.PATCH_LAYERS) if cfg.PATCH_LAYERS else list(range(lm.n_layers))
    D = {q: {k: v.to(lm.device) for k, v in d.items()} for q, d in dirs.items()}
    out, used = [], {k: 0 for k in CF_KINDS}
    for i, pr in enumerate(pairs):
        lb = lm.fwd_last(lm.tensor(pr["ids"]))[0]
        if int(lb.argmax()) != pr["ans"]:
            continue
        allpos = sorted({p for v in pr["P"].values() for p in v})
        for kind, (ids_c, ans_c) in pr["cfs"].items():
            lc, st = lm.run_states(lm.tensor(ids_c), {p: p for p in allpos})
            lc = lc[-1].float()
            if int(lc.argmax()) != ans_c:
                continue
            used[kind] += 1
            a, c = pr["ans"], ans_c
            ldb, ldc = lb[c] - lb[a], lc[c] - lc[a]
            for grp, how in RUNS[kind]:
                posl = pr["P"][grp]
                C = torch.stack([st[p] for p in posl], 1)
                kind_how, qn = (how.split(":") + [None])[:2]
                lp = patched_last(lm, pr["ids"], layers, posl, C, None if qn is None else D[qn][kind_how])
                ld = lp[:, c] - lp[:, a]
                flip = lp.argmax(1) == c
                out += [dict(kind=kind, group=grp, how=kind_how, layer=l, ex=i, flip=bool(flip[j]),
                             nld=float((ld[j] - ldb) / (ldc - ldb))) for j, l in enumerate(layers)]
        if (i + 1) % 50 == 0 or i + 1 == len(pairs):
            log(f"патчинг: {i + 1}/{len(pairs)}")
    log("патчинг: пар в анализе (модель верна в исходном и в контрфакте):", used)
    return pd.DataFrame(out)
