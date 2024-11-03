
# -- import packages: ---------------------------------------------------------
import ABCParse
import matplotlib.pyplot as plt
import pandas as pd


# -- import local dependencies: -----------------------------------------------
from .. import parsers


# -- supporting data operational class: ---------------------------------------
class LossPlotData(ABCParse.ABCParse):
    def __init__(
        self, summarized_loss: parsers.SummarizedLoss, window: int = 50, *args, **kwargs
    ):
        self.__parse__(locals())

    def _smooth(self, values: pd.Series):
        return values.rolling(window=self._window, center=True).mean()

    @property
    def _TRAINING_LOSS(self):
        return self._summarized_loss.training

    @property
    def _VALIDATION_LOSS(self):
        return self._summarized_loss.validation

    @property
    def training(self):
        return {"x": self._TRAINING_LOSS.index, "y": self._TRAINING_LOSS.values}

    @property
    def validation(self):
        return {"x": self._VALIDATION_LOSS.index, "y": self._VALIDATION_LOSS.values}

    @property
    def smoothed_training(self):
        return self._smooth(self._TRAINING_LOSS).dropna()

    @property
    def smoothed_validation(self):
        return self._smooth(self._VALIDATION_LOSS).dropna()


# -- API-facing operational class: ------------------------------------------
class LossPlot(ABCParse.ABCParse):
    def __init__(
        self,
        summarized_loss,
        training_color: str = "#004e89",  # "#00a6ed"
        validation_color: str = "#fbb02d",  # "#ffb400",
        best_epoch_color: str = "#f6511d",
        window: int = 50,
        scatter_kwargs={
            "s": 5,
            "ec": "None",
            "alpha": 0.2,
            "zorder": 10,
            "rasterized": True,
        },
        plot_kwargs={"zorder": 20},
        ymin=500,
        ymax=2500,
    ):
        self.__parse__(locals())

        self.loss_pl_data = LossPlotData(
            summarized_loss=self._summarized_loss, window=self._window
        )

    @property
    def cmap(self):
        return {
            "training": self._training_color,
            "validation": self._validation_color,
            "best": self._best_epoch_color,
        }

    def _plot_raw(self, ax: plt.Axes, key: str):
        loss = getattr(self.loss_pl_data, key)
        ax.scatter(**loss, **self._scatter_kwargs, c=self.cmap[key])

    def _plot_smoothed(self, ax: plt.Axes, key: str):
        smooth_loss = getattr(self.loss_pl_data, f"smoothed_{key}")
        ax.plot(smooth_loss, **self._plot_kwargs, color=self.cmap[key])

    @property
    def best_epoch(self):
        return self._summarized_loss.best_epoch

    @property
    def _name(self):
        seed = str(self._summarized_loss._version).split("_")[-1]
        return f"Seed {seed}"

    def _plot_best_epoch(self, ax):

        ax.vlines(
            x=self.best_epoch,
            ymin=self._ymin,
            ymax=self._ymax,
            ls="--",
            color=self.cmap["best"],
            zorder=20,
        )
        ax.text(
            x=self.best_epoch - 50,
            y=self._ymax - 300,
            s=f"Best epoch: {self.best_epoch}",
            color=self.cmap["best"],
            ha="right",
            fontsize=8,
            zorder=21,
        )

    def _formatting(self, ax, lw: float = 0.5):

        ax.set_xlim(-200, 2700)
        ax.set_ylim(self._ymin, self._ymax)
        ax.grid(alpha=0.2, color="k", zorder=-10)
        [spine.set_lw(lw) for spine in list(ax.spines.values())]
        ax.xaxis.set_tick_params(width=lw)
        ax.yaxis.set_tick_params(width=lw)
        ax.set_title(self._name, fontsize=10)

    def forward(self, ax: plt.Axes, key: str):

        self._plot_raw(ax, key)
        self._plot_smoothed(ax, key)

    def __call__(self, ax: plt.Axes, lw: float = 0.5):
        """ """
        self.__update__(locals())

        self.ax = ax
        self.forward(ax, key="training")
        self.forward(ax, key="validation")
        self._plot_best_epoch(ax)
        self._formatting(ax, lw=lw)


# -- API-facing function: -----------------------------------------------------
def fit_loss(
    summarized_loss: parsers.SummarizedLoss,
    ax: plt.Axes,
    window: int = 50,
    lw: float = 0.5,
    *args,
    **kwargs,
):
    """ """

    loss_plot = LossPlot(
        summarized_loss=summarized_loss,
        window=window,
    )
    loss_plot(ax=ax, lw=lw)
