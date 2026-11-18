"""Опыт probe: что residual stream хранит о роли поля (src/dst), номере слота, указателе на город и уже записанных полях,
и используется ли это (патчинг).

Инструмент с четырьмя полями: два городских (роли src и dst, названия и порядок варьируются) и два посторонних. Модули:
    readout — линейные пробы Q0..Q4 по всем слоям с контролями и отложенными наборами;
    patch   — патчинг пар одной длины (from ↔ to, города местами, названия полей местами) целиком,
              вдоль направления пробы и вдоль случайного направления.

    python main.py --test                    быстрый прогон и проверка всех функций (итог: OK/FAIL)
    python main.py --run                     полный прогон
    python main.py --run --modules readout   только пробы
"""
import argparse, os, random, shutil, sys, time
from dataclasses import replace
import numpy as np, pandas as pd, torch

import analysis, checks, patching, plots, probes
from collect import collect
from config import Config
from data import build_pools, make_dataset, make_spec
from model import LM
from prompts import probe_format
from questions import build_rows, make_questions
from utils import Logger, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """Разбирает флаги командной строки."""
    ap = argparse.ArgumentParser(description="Опыт probe: пробы и патчинг ролей полей инструмента")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true", help="быстрый прогон и проверки всех функций")
    mode.add_argument("--run", action="store_true", help="полный прогон")
    ap.add_argument("--modules", help="через запятую: readout, patch (по умолчанию из config)")
    ap.add_argument("--model", help="переопределить модель (имя на HF или локальная папка)")
    ap.add_argument("--device", help="переопределить устройство: auto, cuda, cpu, mps")
    ap.add_argument("--out", help="папка результатов (по умолчанию results/test или results/run_<время>)")
    return ap.parse_args()


def make_config(args):
    """Собирает настройки для выбранного режима с учётом флагов."""
    cfg = Config.test() if args.test else Config()
    if args.modules:
        cfg = replace(cfg, MODULES=tuple(m.strip() for m in args.modules.split(",")))
    if args.model:
        cfg = replace(cfg, MODEL=args.model)
    if args.device:
        cfg = replace(cfg, DEVICE=args.device)
    if args.out:
        out = os.path.abspath(args.out)
    elif args.test:
        out = os.path.join(HERE, cfg.OUT_DIR)
        shutil.rmtree(out, ignore_errors=True)
    else:
        out = os.path.join(HERE, "results", "run_" + time.strftime("%Y%m%d-%H%M%S"))
    return replace(cfg, OUT_DIR=out)


def main():
    """Данные и состояния, затем выбранные модули; в режиме --test ещё и проверки."""
    args = parse_args()
    cfg = make_config(args)
    log = Logger(cfg.OUT_DIR)
    ck = checks.Checker(log) if args.test else None
    log(f"режим: {'test' if args.test else 'run'}, модули: {', '.join(cfg.MODULES)}, результаты в {cfg.OUT_DIR}")

    set_seed(cfg.SEED)
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(cfg.SEED)

    lm = LM(cfg, log)
    pools = build_pools(lm, log)
    probe_format(lm, make_spec(random.Random(cfg.SEED), pools, "iid"), log)
    if ck:
        checks.check_data(ck, lm, pools)
        checks.check_prompts(ck, lm, pools)
        checks.check_model(ck, lm, pools)
        checks.check_probes(ck, lm.device, cfg)

    specs = make_dataset(cfg, pools, random.Random(cfg.SEED))
    H, meta = collect(lm, specs, log)
    df, lexm = build_rows(lm, specs, meta)
    qs = make_questions(df, cfg)
    log(f"примеры: {len(specs)}, строк: {len(df)}")
    key_rate = analysis.behavior(df, log)
    df.to_csv(f"{cfg.OUT_DIR}/rows.csv", index=False)
    if cfg.SAVE_STATES:
        torch.save(H, f"{cfg.OUT_DIR}/states.pt")
    if ck:
        checks.check_rows(ck, df, lexm, qs, cfg)
        checks.check_planted(ck, lm.device, cfg, df, lexm, qs, H.shape)

    res = pr = None
    if "readout" in cfg.MODULES:
        layers = list(cfg.PROBE_LAYERS) if cfg.PROBE_LAYERS else list(range(lm.n_layers))
        res, cosl = [], []
        for q in qs:
            r, c = probes.run_question(q, df, H, lexm, layers, cfg, rng, lm.device, log)
            res += r; cosl += c
            log(f"{q['name']}: готово (fit {int(q['fit'].sum())}, строк {int(q['base'].sum())})")
        res, cosdf = pd.DataFrame(res), pd.DataFrame(cosl)
        res.to_csv(f"{cfg.OUT_DIR}/probe_results.csv", index=False)
        cosdf.to_csv(f"{cfg.OUT_DIR}/dirs.csv", index=False)
        analysis.summarize(res, log).to_csv(f"{cfg.OUT_DIR}/summary.csv", index=False)
        plots.readout_figures(cfg, res, cosdf, key_rate, log)

    if "patch" in cfg.MODULES:
        dirs = probes.make_dirs(H, df, qs, patching.DIR_QUESTIONS, cfg.SEED)
        pairs = patching.make_pairs(lm, cfg, pools, log)
        if ck:
            checks.check_pairs(ck, lm, pairs)
            checks.check_patch(ck, lm, pairs)
        pr = patching.run_patching(lm, cfg, pairs, dirs, log)
        pr.to_csv(f"{cfg.OUT_DIR}/patch.csv", index=False)
        analysis.summarize_patch(pr, log).to_csv(f"{cfg.OUT_DIR}/patch_summary.csv", index=False)
        plots.patch_figures(cfg, pr, log)

    if ck:
        checks.check_outputs(ck, cfg, res, pr)
    print(shutil.make_archive(cfg.OUT_DIR, "zip", cfg.OUT_DIR))
    log("готово")
    if ck and not ck.report():
        sys.exit(1)


if __name__ == "__main__":
    main()
