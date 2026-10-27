"""Опыт steering: зависит ли успех стиринга от номера аргумента и от семантики значений.

Пять инструментов по пять аргументов, у каждого аргумента своя семантика (имя, город, цвет, профессия и т. д.). Для переходов
1→3, 3→5, 5→3 и 3→1 строится вектор «перенести значение аргумента o в слот аргумента d» по состояниям модели перед записью значений;
проверяется, что его добавка перед слотом o заставляет модель записать значение аргумента d. Вектор каждого инструмента сравнивается
с вектором того же перехода, посчитанным по остальным инструментам, и с контролями.

    python main.py --test    быстрый прогон и проверка всех функций (итог: OK/FAIL)
    python main.py --run     полный прогон

Этапы: модель -> пулы значений -> калибровка -> векторы -> стиринг -> таблицы -> графики.
"""
import argparse, os, shutil, sys, time
from dataclasses import replace
import torch

import analysis, checks, plots, vectors
from config import Config
from data import build_pools
from evaluate import run_all
from model import LM
from prompts import probe_format
from utils import Logger, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """Разбирает флаги командной строки."""
    ap = argparse.ArgumentParser(description="Опыт steering: стиринг между аргументами 1, 3, 5 в инструментах с разной семантикой")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true", help="быстрый прогон и проверки всех функций")
    mode.add_argument("--run", action="store_true", help="полный прогон")
    ap.add_argument("--model", help="переопределить модель (имя на HF или локальная папка)")
    ap.add_argument("--device", help="переопределить устройство: auto, cuda, cpu")
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

    lm = LM(cfg, log)
    pool = build_pools(lm, log)
    probe_format(lm, log)
    if ck:
        checks.check_data(ck, lm, pool)
        checks.check_prompts(ck, lm, pool)
        checks.check_model(ck, lm, cfg, pool)

    H = vectors.collect(lm, cfg, pool, log)
    V = vectors.build_vectors(H)
    vecs = vectors.build_vecs(cfg, V)
    geom = vectors.geometry(cfg, V)
    vectors.save_artifacts(cfg, V, H, pool, geom, log)
    if ck:
        checks.check_vectors(ck, cfg, H, V, vecs)
        checks.check_eval(ck, lm, cfg, vecs, pool)

    df = run_all(lm, cfg, vecs, pool, log)

    res = analysis.analyze(cfg, df, log)
    dd, summary = res if res else (None, None)
    if res:
        plots.make_plots(cfg, summary, log)
        analysis.stats(cfg, dd, log)
    if ck:
        checks.check_outputs(ck, cfg, vecs, df, dd, summary)

    print(shutil.make_archive(cfg.OUT_DIR, "zip", cfg.OUT_DIR))
    log("готово")
    if ck and not ck.report():
        sys.exit(1)


if __name__ == "__main__":
    main()
