# -- import packages: ---------------------------------------------------------
import numpy as np

# -- set classes: -------------------------------------------------------------
class FitLine:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    @property
    def coeffs(self) -> np.ndarray:
        return np.polyfit(self.x, self.y, 1)

    @property
    def slope(self) -> float:
        return self.coeffs[0]

    @property
    def intercept(self) -> float:
        return self.coeffs[1]

    @property
    def y_fit(self) -> np.ndarray:
        return self.slope * self.x + self.intercept
