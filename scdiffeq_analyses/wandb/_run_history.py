# -- import packages: ---------------------------------------------------------
import pandas as pd
import pathlib

# -- import local dependencies: -----------------------------------------------
from .. import types
from ._download_run_history import download_run_history

# -- operational object cls: --------------------------------------------------
class RunHistory:
    def __init__(self, run: types.Run) -> None:
        self.run = run

    @property
    def history_path(self) -> pathlib.Path:
        return self.run.dir.joinpath("history.csv")

    @property
    def df(self) -> pd.DataFrame:
        if self.run.state == "finished" and (not hasattr(self, "_history")):
            if not self.history_path.exists():
                self._history = download_run_history(
                    run=self.run._run, fpath=self.history_path
                )
            else:
                self._history = pd.read_csv(self.history_path, index_col=0)
        else:
            self._history = None
        return self._history
