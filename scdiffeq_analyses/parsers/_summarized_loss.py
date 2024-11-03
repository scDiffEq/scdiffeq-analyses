# -- import packages: ---------------------------------------------------------
import ABCParse
import pandas as pd
import scdiffeq as sdq


# -- supporting class: --------------------------------------------------------
class PerEpochLoss(ABCParse.ABCParse):
    """Tabulator of per epoch loss"""

    def __init__(
        self,
        t_ignore="0.0",
        epoch_key: str = "epoch",
        metric: str = "sinkhorn",
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            t_ignore (str) ***Default**: "0.0"

            epoch_key (str) ***Default**: "epoch"

            metric (str) ***Default**: "sinkhorn"

        Returns:
            None
        """
        self.__parse__(locals())

    def _aggregate(self, df: pd.DataFrame):
        """Filter, sum, mean: apply, per epoch to grouped df"""
        return (
            df.filter(regex=self._stage)
            .filter(regex=self._metric)
            .drop(f"{self._metric}_{self._t_ignore}_{self._stage}", axis=1)
            .dropna()
            .sum(1)
            .mean()
        )

    @property
    def _EPOCH_GROUPED_METRICS(self):
        """ """
        if not hasattr(self, "_GROUPED_BY"):
            self._GROUPED_BY = self._metrics_df.groupby(self._epoch_key)
        return self._GROUPED_BY

    def forward(self):
        """ """
        return self._EPOCH_GROUPED_METRICS.apply(self._aggregate) # , include_groups=False)

    def __call__(
        self, metrics_df: pd.DataFrame, stage: str = "training", *args, **kwargs
    ) -> pd.Series:
        """
        Args:
            metrics_df (pd.DataFrame)

            stage (str)
        Returns:
            loss (pd.Series)
        """
        self.__update__(locals())

        return self.forward()


# -- operational class: -------------------------------------------------------
class SummarizedLoss(ABCParse.ABCParse):
    def __init__(
        self,
        version: sdq.io.Version,
        n_best: int = 10,
        training_key: str = "training",
        validation_key: str = "validation",
        t_ignore="0.0",
        epoch_key: str = "epoch",
        metric: str = "sinkhorn",
        *args,
        **kwargs
    ):
        self.__parse__(locals())

    @property
    def metrics_df(self) -> pd.DataFrame:
        if not hasattr(self, "_metrics_df"):
            self._metrics_df = self._version.metrics_df
        return self._metrics_df

    @property
    def _PER_EPOCH_LOSS_TABULATOR(self):
        if not hasattr(self, "_per_epoch_loss_tabulator"):
            self._per_epoch_loss_tabulator = PerEpochLoss(
                epoch_key=self._epoch_key, metric=self._metric, t_ignore=self._t_ignore,
            )
        return self._per_epoch_loss_tabulator

    @property
    def training(self) -> pd.Series:
        if not hasattr(self, "_training"):
            self._training = self._PER_EPOCH_LOSS_TABULATOR(
                self.metrics_df, stage=self._training_key
            )
        return self._training

    @property
    def validation(self) -> pd.Series:
        if not hasattr(self, "_validation"):
            self._validation = self._PER_EPOCH_LOSS_TABULATOR(
                self.metrics_df, stage=self._validation_key
            )
        return self._validation

    @property
    def best_validation(self):
        if not hasattr(self, "_best_validation"):
            self._best_validation = (
                self.validation.sort_values()
                .head(self._n_best)
                .to_frame()
                .rename({0: "validation_loss"}, axis=1)
            )
        return self._best_validation

    @property
    def _SAVED_CKPTS(self):
        if not hasattr(self, "_saved_ckpts"):
            self._saved_ckpts = list(self._version.ckpts.keys())
        return self._saved_ckpts

    @property
    def _BEST_SAVED_CKPTS(self):
        if not hasattr(self, "_best_saved_ckpts"):
            self._best_saved_ckpts = self.best_validation.loc[
                self.best_validation.index.isin(self._SAVED_CKPTS)
            ]
        return self._best_saved_ckpts

    @property
    def best_epoch(self):
        """best validation epoch"""
        if not hasattr(self, "_best_epoch"):
            self._best_epoch = self.validation.loc[
                self._BEST_SAVED_CKPTS.index
            ].idxmax()
        return self._best_epoch

    @property
    def best_ckpt(self) -> sdq.io.Checkpoint:
        """ """
        if not hasattr(self, "_best_ckpt"):
            self._best_ckpt = self._version.ckpts[self.best_epoch]
        return self._best_ckpt
    
    def __repr__(self) -> str:
        return f"""SummarizedLoss[{self._version._NAME}]"""
    