# -- import packages: ---------------------------------------------------------
import numpy as np

# -- cls: ---------------------------------------------------------------------
class ForegroundScatter:
    def __init__(self, colors, scatter_kw) -> None:
        """Foreground scatter points."""
        self.colors = colors
        self.scatter_kw = scatter_kw

    def forward(self, en, key, val) -> None:

        if self._use_x:
            x = np.full(len(val), key)
            if len(x) > 1 and self._jitter:
                x_vals = x + (np.random.random(len(x)) - 0.5) / 5
            else:
                x_vals = x
        else:
            x = [key] * len(val)
            if len(x) > 1 and self._jitter:
                x_vals = en + 1 + (np.random.random(len(x)) - 0.5) / 5
            else:
                x_vals = en + 1

        self.ax.scatter(
            x_vals,
            val,
            color=self.colors[en],
            zorder=0,
            ec="None",
            rasterized=False,
            **self.scatter_kw,
        )

    def __call__(self, ax, data, use_x: bool = False, jitter: bool = True) -> None:

        self.ax = ax
        self._jitter = jitter
        self._use_x = use_x

        for en, (key, val) in enumerate(data.items()):
            self.forward(en, key, val)
