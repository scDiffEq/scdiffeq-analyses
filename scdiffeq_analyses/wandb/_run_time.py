# -- import packages: ---------------------------------------------------------
import pandas as pd

# -- import local dependencies: -----------------------------------------------
from ._run import Run

# -- set type hints: ----------------------------------------------------------
from typing import List

# -- set constants: -----------------------------------------------------------
RUNTIME_COLS = ["epoch", "_timestamp", "_runtime"]

# -- operational object cls: --------------------------------------------------
class RunTime:
    def __init__(self, run, cols: List[str] = RUNTIME_COLS) -> None:
        """"""
        self.run = run
        self._cols = cols

    @property
    def t_history(self):
        if not hasattr(self, "_t_history"):
            self._t_history = self.run.history[self._cols].dropna().copy()
        return self._t_history

    @property
    def total(self):
        """in seconds.
        includes setup (i.e., the time between zero and the
        first timestamp in history"""
        return self.run._run.summary["_runtime"]

    @property
    def training(self):
        """Just training, ignores data-related setup, etc."""
        t_init = self.t_history["_runtime"].iloc[0]
        t_final = self.t_history["_runtime"].iloc[-1]
        return t_final - t_init

    @property
    def per_epoch(self) -> pd.Series:
        """time for each epoch"""
        return self.t_history.groupby("epoch")["_runtime"].max().diff().dropna()

    @property
    def per_epoch_mean(self):
        """in seconds"""
        return self.per_epoch.mean()
