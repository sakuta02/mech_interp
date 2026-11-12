"""Параметры опыта в одном месте: размеры выборок, пробы, слои, модель и устройство."""
from dataclasses import dataclass, replace


@dataclass
class Config:
    """Параметры опыта: при настройке эксперимента правятся здесь."""
    MODEL: str = "openbmb/MiniCPM5-2B"
    DEVICE: str = "auto"                  # auto: cuda, если есть, иначе cpu
    SEED: int = 0
    N_IID: int = 2000                     # примеров из обычного распределения; делятся на fit / val / test по примерам
    N_OOD: int = 300                      # примеров на каждый отложенный набор (города, названия, описания, шаблоны, всё сразу)
    N_SWAP: int = 500                     # примеров swap-fill: первое городское поле заполнено городом другой роли
    FIT_FRAC: tuple = (0.7, 0.15)         # доли fit и val среди iid; остальное test
    ONLY_CORRECT: bool = True             # в fit и в оценку идут примеры, где модель сама верна в обоих городских слотах
    PROBE_LAYERS: tuple = None            # None: все слои 0..n_layers-1; иначе только перечисленные
    L2_GRID: tuple = (1e-3, 1e-2, 1e-1)   # сила L2 выбирается по val
    LBFGS_ITERS: int = 100
    SHUF_SEEDS: int = 1                   # сколько раз обучать контроль с перемешанными метками
    BOOT: int = 200                       # бутстреп-повторов для доверительных интервалов (по примерам)
    SAVE_STATES: bool = False             # сохранять ли все состояния (несколько ГБ)
    OUT_DIR: str = "results/run"

    @classmethod
    def test(cls):
        """Урезанные параметры для быстрого прогона --test."""
        return replace(cls(), N_IID=120, N_OOD=16, N_SWAP=16, PROBE_LAYERS=(0, 10, 20, 30, 41),
                       L2_GRID=(1e-2, 1e-1), LBFGS_ITERS=30, BOOT=20, OUT_DIR="results/test")
