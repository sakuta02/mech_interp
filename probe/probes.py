"""Пробы: логистическая регрессия (L-BFGS, как рекомендуют Статьи), ближайший центроид (разность средних), контроли
(перемешанные метки, бейзлайны по позиции и по названиям полей) и метрики с бутстреп-интервалами по примерам."""
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy.stats import rankdata


def fit_lr(X, y, ncls, l2, iters):
    """Многоклассовая логистическая регрессия с весами классов и L2; возвращает (W [d, C], b [C])."""
    W = torch.zeros(X.shape[1], ncls, device=X.device, requires_grad=True)
    b = torch.zeros(ncls, device=X.device, requires_grad=True)
    cnt = torch.bincount(y, minlength=ncls).float()
    w = torch.where(cnt > 0, len(y) / (ncls * cnt.clamp(min=1)), torch.zeros_like(cnt))
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        """Взвешенная кросс-энтропия плюс L2 по весам."""
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y, weight=w) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    return W.detach(), b.detach()


def centroid_logits(Xf, yf, X, ncls):
    """Логиты ближайшего центроида: −‖x − μ_c‖² (для двух классов это проекция на разность средних); μ_c по fit."""
    mu = torch.stack([Xf[yf == c].mean(0) if (yf == c).any() else torch.full_like(Xf[0], 1e6) for c in range(ncls)])
    return -torch.cdist(X, mu) ** 2, mu


def bacc(y, pred, ncls):
    """Сбалансированная точность: средний recall по классам, которые есть в y."""
    r = [(pred[y == c] == c).mean() for c in range(ncls) if (y == c).any()]
    return float(np.mean(r)) if r else np.nan


def auroc(y, s):
    """AUROC для двух классов через ранги; NaN, если один из классов отсутствует."""
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(y, pred, g, ncls, B, rng):
    """95% интервал сбалансированной точности: бутстреп по группам g (примерам), а не по строкам."""
    _, inv = np.unique(g, return_inverse=True)
    G = inv.max() + 1
    cnt, hit = np.zeros((G, ncls)), np.zeros((G, ncls))
    np.add.at(cnt, (inv, y), 1); np.add.at(hit, (inv, y), pred == y)
    idx = rng.integers(0, G, (B, G))
    c, h = cnt[idx].sum(1), hit[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.nanmean(np.where(c > 0, h / c, np.nan), 1)
    return float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


def score(logits, ncls):
    """Скалярная оценка для AUROC: разность логитов двух классов."""
    return (logits[:, 1] - logits[:, 0]) if ncls == 2 else None


def evaluate(logits, y, g, ncls, cfg, rng, ci=True):
    """Метрики одного набора: bacc, auroc, доверительный интервал, n; NaN, если в наборе один класс."""
    if len(np.unique(y)) < 2:
        return dict(n=len(y), bacc=np.nan, auroc=np.nan, lo=np.nan, hi=np.nan)
    pred = logits.argmax(1).cpu().numpy()
    s = score(logits, ncls)
    lo, hi = boot_ci(y, pred, g, ncls, cfg.BOOT, rng) if ci and len(y) > 1 else (np.nan, np.nan)
    return dict(n=len(y), bacc=bacc(y, pred, ncls), auroc=auroc(y, s.cpu().numpy()) if s is not None else np.nan, lo=lo, hi=hi)


def probe_suite(X, y, fit, val, evals, g, ncls, cfg, rng, std=True, extra=True):
    """Обучает пробы на fit и оценивает на наборах evals {имя: индексы}; y — метки всех строк X (torch), g — примеры (numpy).
    L2 выбирается по val. Возвращает (список строк результата, cos между направлениями логрега и центроида или None)."""
    if len(fit) == 0 or len(torch.unique(y[fit])) < 2:
        return [], None
    mu, sd = (X[fit].mean(0), X[fit].std(0) + 1e-6) if std else (0, 1)
    Z = (X - mu) / sd
    best = max(cfg.L2_GRID, key=lambda a: _val_score(Z, y, fit, val, ncls, a, cfg))
    W, b = fit_lr(Z[fit], y[fit], ncls, best, cfg.LBFGS_ITERS)
    out, yn = [], y.cpu().numpy()

    def add(kind, logits_fn, ci=True):
        """Считает метрики пробы на всех наборах."""
        for name, (ix, ycol) in evals.items():
            if len(ix) == 0:
                continue
            yy = yn[ix] if ycol is None else ycol[ix]
            out.append(dict(probe=kind, set=name, l2=best, **evaluate(logits_fn(Z[ix]), yy, g[ix], ncls, cfg, rng, ci)))
    add("logreg", lambda Zi: Zi @ W + b)
    if not extra:
        return out, None
    _, mu_c = centroid_logits(Z[fit], y[fit], Z[:1], ncls)
    add("dim", lambda Zi: centroid_logits(Z[fit], y[fit], Zi, ncls)[0], ci=False)
    cos = None
    if ncls == 2:
        cos = float(F.cosine_similarity(mu_c[1] - mu_c[0], W[:, 1] - W[:, 0], dim=0))
    ys = y[fit].cpu().numpy()
    for s in range(cfg.SHUF_SEEDS):
        perm = torch.as_tensor(rng.permutation(ys), device=X.device)
        Ws, bs = fit_lr(Z[fit], perm, ncls, best, cfg.LBFGS_ITERS)
        add(f"shuf{s}", lambda Zi: Zi @ Ws + bs, ci=False)
    return out, cos


def _val_score(Z, y, fit, val, ncls, l2, cfg):
    """Сбалансированная точность логрега с силой l2 на val (по fit, если val пуст)."""
    W, b = fit_lr(Z[fit], y[fit], ncls, l2, cfg.LBFGS_ITERS)
    ix = val if len(val) else fit
    return bacc(y[ix].cpu().numpy(), (Z[ix] @ W + b).argmax(1).cpu().numpy(), ncls)


def onehot_pos(df_rows, tmax):
    """Бейзлайн по позиции: one-hot только абсолютной позиции токена (номер позиции чтения нельзя: он совпадает с метками Q0 и Q2)."""
    p = np.zeros((len(df_rows), tmax + 1), np.float32)
    p[np.arange(len(df_rows)), df_rows.pos_abs.values] = 1
    return p


def run_question(q, df, H, lexm, layers, cfg, rng, device, log):
    """Все слои одного вопроса: пробы по состояниям модели плюс бейзлайны (layer = -1). Возвращает (результаты, cos по слоям)."""
    b = np.flatnonzero(q["base"].values)
    if len(b) == 0:
        return [], []
    pos = {i: j for j, i in enumerate(b)}
    sub = df.iloc[b]
    y = torch.as_tensor(sub[q["y"]].values.astype(np.int64), device=device)
    g = sub.ex.values

    def ix(mask):
        """Позиции строк маски внутри строк вопроса."""
        return np.array([pos[i] for i in np.flatnonzero(mask.values) if i in pos], dtype=np.int64)
    fit, val = ix(q["fit"]), ix(q["val"])
    ev = {}
    for name, (m, ycol) in q["evals"].items():
        ev[name] = (ix(m), None if ycol == q["y"] else sub[ycol].values.astype(np.int64))
    res, cosl = [], []

    def tag(rows, layer):
        """Добавляет к строкам результата имя вопроса и слой."""
        return [dict(question=q["name"], layer=layer, **r) for r in rows]
    ex_idx, site_idx = torch.as_tensor(sub.ex.values.copy()), torch.as_tensor(sub.site.values.copy())
    for l in layers:
        X = H[l][ex_idx, site_idx].to(device).float()
        rows, cos = probe_suite(X, y, fit, val, ev, g, q["ncls"], cfg, rng)
        res += tag(rows, l)
        if cos is not None:
            cosl.append(dict(question=q["name"], layer=l, cos=cos))
    tmax = int(df.pos_abs.max())
    feats = {"pos": torch.as_tensor(onehot_pos(sub, tmax), device=device)}
    if q["lex"]:
        feats["lex"] = torch.as_tensor(lexm[b], device=device)
    for kind, X in feats.items():
        rows, _ = probe_suite(X, y, fit, val, ev, g, q["ncls"], cfg, rng, std=False, extra=False)
        res += tag([dict(r, probe=kind) for r in rows], -1)
    return res, cosl
