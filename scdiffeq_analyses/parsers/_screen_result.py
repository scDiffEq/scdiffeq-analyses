# -- import packages: ---------------------------------------------------------
import pandas as pd

# -- operational object cls: --------------------------------------------------
class ScreenResult:
    """Fate perturbation screen result (singular version)"""
    def __init__(self, base_path: str) -> None:
        self._base_path = base_path
        self._suffixes = {
            "lfc_mean": ".lfc.csv",
            "pval": ".pval.csv",
            "lfc_std": ".lfc_std.csv",
        }

    def _fetch_data(self, accessor: str) -> pd.DataFrame:
        if not hasattr(self, f"_{accessor}"):
            csv_path = str(self._base_path) + self._suffixes[accessor]
            df = pd.read_csv(csv_path, index_col=0)
            setattr(self, f"_{accessor}", df)
        return getattr(self, f"_{accessor}")

    @property
    def lfc_mean(self) -> pd.DataFrame:
        return self._fetch_data("lfc_mean")

    @property
    def pval(self) -> pd.DataFrame:
        return self._fetch_data("pval")

    @property
    def lfc_std(self) -> pd.DataFrame:
        return self._fetch_data("lfc_std")
