# -- import packages: ---------------------------------------------------------
import pandas as pd

# -- set type hints: ----------------------------------------------------------
from typing import List

# -- set constants: -----------------------------------------------------------
MEMORY_COLUMNS = [
    "epoch",
    "Memory/GPU_Reserved_MB",
    "Memory/GPU_Allocated_MB",
    "Memory/CPU_RAM_MB",
]

# -- operational object cls: --------------------------------------------------
class RunMemory:
    def __init__(
        self,
        history_df: pd.DataFrame,
        columns: List[str] = MEMORY_COLUMNS,
    ) -> None:
        """"""

        self._history_df = history_df
        self._columns = columns

    @property
    def df(self) -> pd.DataFrame:
        if not hasattr(self, "_df"):
            mem_df = self._history_df[self._columns].copy()
            mem_df["epoch"] = mem_df["epoch"].ffill()
            self._df = mem_df
        return self._df

    @property
    def per_epoch_max(self) -> pd.DataFrame:
        if not hasattr(self, "_per_epoch_max"):
            self._per_epoch_max = self.df.groupby("epoch").max()
        return self._per_epoch_max

    @property
    def max_requirements(self) -> pd.Series:
        if not hasattr(self, "_max_requirements"):
            self._max_requirements = self.per_epoch_max.max()
        return self._max_requirements

    def __repr__(self) -> str:
        return "RunMemory"
