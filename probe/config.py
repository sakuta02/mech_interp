"""Параметры опыта в одном месте: модули, размеры выборок, пробы, патчинг, модель и устройство."""
from dataclasses import dataclass, replace


@dataclass
class Config:
    """Параметры опыта: при настройке эксперимента правятся здесь."""
    MODEL: str = "openbmb/MiniCPM5-2B"
    DEVICE: str = "auto"                  # auto: cuda, если есть, иначе cpu
    SEED: int = 0
    MODULES: tuple = ("readout", "patch")   # какие модули опыта запускать: пробы и патчинг
    # выборки
    N_IID: int = 2000                     # обычное распределение; делится на fit / val / test по примерам
    N_OOD: int = 300                      # на каждый отложенный набор (города, названия, описания, шаблоны, всё сразу)
    N_SWAP: int = 500                     # первое городское поле заполнено городом другой роли
    N_POSTFIX: int = 1000                 # только postfix-шаблоны: контроль на утечку Q0_leak
    N_CONFLICT: int = 300                 # описание городского поля взято у другой роли
    FIT_FRAC: tuple = (0.7, 0.15)         # доли fit и val; остальное test
    ONLY_CORRECT: bool = True             # в fit и в основную оценку идут примеры, где модель сама верна в городских слотах
    # пробы
    PROBE_LAYERS: tuple = None            # None: все слои 0..n_layers-1
    L2_GRID: tuple = (1e-3, 1e-2, 1e-1)   # сила L2 выбирается по val
    LBFGS_ITERS: int = 100
    SHUF_SEEDS: int = 1                   # сколько раз обучать контроль с перемешанными метками
    BOOT: int = 200                       # бутстреп-повторов (по примерам)
    # патчинг
    N_PATCH: int = 300                    # примеров; у каждого три контрфакта (marker, city, schema)
    PATCH_LAYERS: tuple = None            # None: все слои
    SAVE_STATES: bool = False             # сохранять ли все состояния (несколько ГБ)
    OUT_DIR: str = "results/run"

    @classmethod
    def test(cls):
        """Урезанные параметры для быстрого прогона --test."""
        return replace(cls(), N_IID=120, N_OOD=16, N_SWAP=16, N_POSTFIX=40, N_CONFLICT=16,
                       PROBE_LAYERS=(0, 10, 20, 30, 41), L2_GRID=(1e-2, 1e-1), LBFGS_ITERS=30, BOOT=20,
                       N_PATCH=4, PATCH_LAYERS=(0, 10, 20, 30, 41), OUT_DIR="results/test")
