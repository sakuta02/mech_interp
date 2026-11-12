"""Опыт probe: хранит ли residual stream смысл поля (src/dst), номер слота, указатель на город и учёт уже записанных городов.

Инструмент с четырьмя полями: два городских (роли src и dst, названия и порядок варьируются) и два посторонних. На одном
teacher-forced проходе читаются состояния всех слоёв в слотах и на токенах городов; затем обучаются линейные пробы Q0..Q4 с контролями
(перемешанные метки, бейзлайны по позиции и по названиям полей, отложенные города, названия, описания и шаблоны).

    python main.py --test    быстрый прогон и проверка всех функций (итог: OK/FAIL)
    python main.py --run     полный прогон

Этапы: модель -> пулы -> примеры -> сбор состояний -> строки и вопросы -> пробы -> сводка -> графики.
"""
import argparse, os, random, shutil, sys, time
from dataclasses import replace
import numpy as np, pandas as pd, torch

import analysis, checks, plots, probes
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
    ap = argparse.ArgumentParser(description="Опыт probe: линейные пробы роли поля, номера слота, указателя и учёта записанных городов")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true", help="быстрый прогон и проверки всех функций")
    mode.add_argument("--run", action="store_true", help="полный прогон")
    ap.add_argument("--model", help="переопределить модель (имя на HF или локальная папка)")
    ap.add_argument("--device", help="переопределить устройство: auto, cuda, cpu, mps")
    ap.add_argument("--out", help="папка результатов (по умолчанию results/test или results/run_<время>)")
    return ap.parse_args()


def make_config(args):
    """Собирает настройки для выбранного режима с учётом флагов."""
    cfg = Config.test() if args.test else Config()
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
    """Запускает все этапы опыта; в режиме --test ещё и проверки."""
    args = parse_args()
    cfg = make_config(args)
    log = Logger(cfg.OUT_DIR)
    ck = checks.Checker(log) if args.test else None
    log(f"режим: {'test' if args.test else 'run'}, результаты в {cfg.OUT_DIR}")

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
    log(f"примеры: {len(specs)}, строк: {len(df)}, верных моделью (iid): {df[df.group == 'iid'].groupby('ex').correct.first().mean():.2f}")
    df.to_csv(f"{cfg.OUT_DIR}/rows.csv", index=False)
    if cfg.SAVE_STATES:
        torch.save(H, f"{cfg.OUT_DIR}/states.pt")
    if ck:
        checks.check_rows(ck, df, lexm, qs, cfg)
        checks.check_planted(ck, lm.device, cfg, df, lexm, qs, H.shape)

    layers = list(cfg.PROBE_LAYERS) if cfg.PROBE_LAYERS else list(range(lm.n_layers))
    res, cosl = [], []
    for q in qs:
        r, c = probes.run_question(q, df, H, lexm, layers, cfg, rng, lm.device, log)
        res += r; cosl += c
        log(f"{q['name']}: готово (fit {int(q['fit'].sum())}, строк {int(q['base'].sum())})")
    res = pd.DataFrame(res)
    res.to_csv(f"{cfg.OUT_DIR}/probe_results.csv", index=False)
    pd.DataFrame(cosl).to_csv(f"{cfg.OUT_DIR}/dirs.csv", index=False)
    summ = analysis.summarize(res, log)
    summ.to_csv(f"{cfg.OUT_DIR}/summary.csv", index=False)
    plots.make_plots(cfg, res, log)
    if ck:
        checks.check_outputs(ck, cfg, res, summ)

    print(shutil.make_archive(cfg.OUT_DIR, "zip", cfg.OUT_DIR))
    log("готово")
    if ck and not ck.report():
        sys.exit(1)


if __name__ == "__main__":
    main()
