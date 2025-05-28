# -- import packages: ---------------------------------------------------------
import pandas as pd

# -- import local dependencies: -----------------------------------------------
from ._run import Run

class Runtime:
    def __init__(self, run: Run) -> None:
        self._run = run
        self._history = self._run.history[["epoch", "_timestamp", "_runtime"]].dropna()

    @property
    def total(self):
        """in seconds.
        includes setup (i.e., the time between zero and the
        first timestamp in history"""
        return self._run._run.summary["_runtime"]

    @property
    def training(self):
        """Just training, ignores data-related setup, etc."""
        t_init = self._history["_runtime"].iloc[0]
        t_final = self._history["_runtime"].iloc[-1]
        return t_final - t_init

    @property
    def per_epoch(self) -> pd.Series:
        """time for each epoch"""
        return self._history.groupby("epoch")["_runtime"].max().diff().dropna()

    @property
    def per_epoch_mean(self):
        """in seconds"""
        return self.per_epoch.mean()
