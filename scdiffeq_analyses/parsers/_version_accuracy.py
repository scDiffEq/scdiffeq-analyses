# -- import packages: ---------------------------------------------------------
import ABCParse
import pandas as pd
import pathlib
import scdiffeq as sdq


# -- set typing: --------------------------------------------------------------
from typing import List


# -- operational object class: ------------------------------------------------
class VersionAccuracy(ABCParse.ABCParse):
    def __init__(
        self,
        version: sdq.io.Version,
        train_col: str = "unique_train.all_fates",
        test_col: str = "unique_test.all_fates",
        *args,
        **kwargs,
    ):
        self.__parse__(locals())

    @property
    def csv_paths(self) -> List[pathlib.Path]:
        return list(self._version._PATH.glob("fate_prediction_metrics/*/accuracy.csv"))

    def _read_frame(self, path: pathlib.Path):

        name = path.parent.name
        return (
            pd.read_csv(path, index_col=0)
            .loc[[self._train_col, self._test_col]]
            .rename({self._train_col: "train", self._test_col: "test"}, axis=0)
            .rename({"accuracy": name}, axis=1)
        )

    def _format_ckpt_name(self, name):
        return name.replace("=", "_").split(".ckpt")[0].replace("-", ".")

    @property
    def _saved_ckpt_fpaths(self) -> pathlib.Path:
        return [ckpt.path for ckpt in self._version.ckpts.values()]

    @property
    def _saved_ckpts(self):
        """Not all (e.g., `on_train_epoch_end`) ckpts get saved. Use this to filter accordingly"""
        return [self._format_ckpt_name(ckpt.name) for ckpt in self._saved_ckpt_fpaths]

    def _get_ckpt_epoch_values(self, df):
        ckpt_epochs = df.index.str.split(".", expand=True).to_frame()[0].values.tolist()
        epochs = []
        for epoch in ckpt_epochs:
            if epoch == "last":
                epochs.append(2500)
            else:
                epochs.append(int(epoch.split("epoch_")[1]))

        return epochs

    @property
    def df(self):
        if not hasattr(self, "_df"):
            df = pd.concat(
                [self._read_frame(path) for path in self.csv_paths], axis=1
            ).T
            df = df.loc[df.index.isin(self._saved_ckpts)]
            df["ckpt_path"] = self._saved_ckpt_fpaths
            df["epoch"] = self._get_ckpt_epoch_values(df)
            self._df = df.sort_values(["train", "epoch"], ascending=[False, False])
        return self._df

    @property
    def best_training_ckpt(self):
        return self.df.index[0]

    @property
    def best_test_from_train(self) -> pd.Series:
        return self.df.loc[self.best_training_ckpt]
