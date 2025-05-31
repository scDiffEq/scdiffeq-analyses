import numpy as np
import matplotlib.cm as cm
class BoxPlot:
    def __init__(self, boxplot_kwargs: dict, colors: list = cm.tab20.colors) -> None:

        self.boxplot_kwargs = boxplot_kwargs
        self.colors = colors
    @property
    def y(self):
        return list(self.data.values())

    @property
    def x(self):
        if self._use_x:
            return np.array(list(self.data.keys()))
        else:
            return np.arange(len(self.y)) + 1

    def forward(self):
        bp = self.ax.boxplot(
            self.y,
            positions=self.x,
            patch_artist=True,
            **self.boxplot_kwargs,
        )
        if self.mode == "background":
            bp = self._format_background(bp)
        elif self.mode == "foreground":
            bp = self._format_foreground(bp)

        bp = self._format_single_data_point(bp)

        return bp

    def _format_background(self, bp):

        for median in bp["medians"]:
            median.set_visible(False)
        for en, mean in enumerate(bp["means"]):
            mean.set_c(self.colors[en])

        for en, box in enumerate(bp["boxes"]):
            box.set_facecolor(self.colors[en])
            box.set_alpha(0.2)

        for en, whisker in enumerate(bp["whiskers"]):
            whisker.set_c("None")

        for en, cap in enumerate(bp["caps"]):
            cap.set_c("None")

        return bp

    def _format_foreground(self, bp):
        for en, box in enumerate(bp["boxes"]):
            box.set_facecolor("None")
            box.set_edgecolor(self.colors[en])

        colors_ = np.repeat(
            np.array(self.colors), 2, axis=0
        )  # list(np.repeat(self.colors, 2))
        for en, whisker in enumerate(bp["whiskers"]):
            whisker.set_c(colors_[en])

        for en, cap in enumerate(bp["caps"]):
            cap.set_c(colors_[en])

        for median in bp["medians"]:
            median.set_visible(False)

        return bp

    def _format_single_data_point(self, bp):
        # Remove boxes for categories with single data points
        for en, (key, val) in enumerate(self.data.items()):
            if len(val) == 1:
                for bp_key in bp.keys():
                    if len(bp[bp_key]) > 0:
                        if len(bp[bp_key]) == len(self.data):
                            bp[bp_key][en].set_visible(False)
                        elif len(bp[bp_key]) == len(self.data) * 2:
                            bp[bp_key][2 * en].set_visible(False)
                            bp[bp_key][2 * en + 1].set_visible(False)
        return bp

    def __call__(
        self,
        ax,
        data,
        use_x: bool = False,
        mode: str = "background",
    ):

        self.ax = ax
        self.data = data
        self.mode = mode
        self._use_x = use_x

        return self.forward()
