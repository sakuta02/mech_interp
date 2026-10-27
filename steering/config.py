"""Параметры опыта в одном месте: размеры выборок, силы вмешательства, диапазон слоёв, модель и устройство."""
from dataclasses import dataclass, replace


@dataclass
class Config:
    """Параметры опыта: при настройке эксперимента правятся здесь."""
    MODEL: str = "openbmb/MiniCPM5-2B"
    DEVICE: str = "auto"                  # auto: cuda, если есть, иначе cpu
    SEED: int = 0
    N_PER_TOOL: int = 72                  # примеров калибровки на инструмент
    N_TEST: int = 600                     # примеров стиринга на рейсах
    N_TEST_DOM: int = 120                 # примеров на каждый новый тип задач
    ALPHAS: tuple = (1.5, 2.0, 3.0, 4.0)
    CTRL_ALPHAS: tuple = (2.0, 3.0, 4.0)  # силы для контролей
    LAYER_RANGE: range = range(18, 28)
    L0: int = 23                          # слой и сила для таблиц и тестов
    A0: float = 3.0
    N_CITIES: int = 60                    # пул городов; делится пополам на калибровку и тест
    BATCH: int = 40
    FILTER_CALIB: bool = True             # в вектор идут только примеры, где модель сама верна в обоих слотах
    OUT_DIR: str = "results/run"

    @classmethod
    def test(cls):
        """Урезанные параметры для быстрого прогона --test."""
        return replace(cls(), N_PER_TOOL=6, N_TEST=8, N_TEST_DOM=4, ALPHAS=(3.0,), CTRL_ALPHAS=(3.0,),
                       LAYER_RANGE=range(22, 24), L0=23, A0=3.0, BATCH=4, FILTER_CALIB=False,
                       OUT_DIR="results/test")
