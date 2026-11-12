"""Сбор состояний: один teacher-forced проход на пример, resid_pre всех слоёв читается хуками в слотах и на токенах городов."""
import torch
from data import N_SLOTS
from prompts import call_ids

SITES = [f"slot{k}" for k in range(1, N_SLOTS + 1)] + ["city_src", "city_dst"]


def collect(lm, specs, log):
    """Возвращает H [L, N, len(SITES), d] (fp16, cpu) и мета по примерам: позиции, длина, argmax модели в каждом слоте."""
    N, d = len(specs), lm.model.cfg.d_model
    H = torch.zeros(lm.n_layers, N, len(SITES), d, dtype=torch.float16)
    meta = []
    for i, sp in enumerate(specs):
        ids, pos, cpos = call_ids(lm, sp)
        sites = {**{f"slot{k}": pos[k] for k in pos}, "city_src": cpos["src"], "city_dst": cpos["dst"]}
        logits, st = lm.run_states(lm.tensor(ids), sites)
        H[:, i] = torch.stack([st[s] for s in SITES], 1).half().cpu()
        meta.append(dict(pos=[pos[k] for k in range(1, N_SLOTS + 1)], cpos=cpos, seq_len=len(ids),
                         pred=[int(logits[pos[k]].argmax()) for k in range(1, N_SLOTS + 1)]))
        if (i + 1) % 250 == 0 or i + 1 == N:
            log(f"собрано {i + 1}/{N}")
    return H, meta
