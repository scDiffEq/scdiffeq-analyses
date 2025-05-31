# -- import packages: ---------------------------------------------------------
import pandas as pd

# -- import local dependencies: -----------------------------------------------
from ._track_completion import track_completion

# -- set type hints: ----------------------------------------------------------
from typing import Dict, List, Tuple

# -- operational cls: ---------------------------------------------------------
class PlotDataAggr:
    """Hyperparameter scan plotting data"""

    def __init__(self, summary_df: pd.DataFrame) -> None:
        """"""
        self._summary_df = summary_df
        self._plot_data = {}
        self._fetch_all_plot_data()

    def print(self) -> None:
        for k, v in self._plot_data.items():
            print(f"{k:<15} {len(v)}")

    @property
    def tracked_completion(self) -> dict:
        if not hasattr(self, "_tracked_completion"):
            (
                self._tracked_completion,
                self._condition_list_set,
                self._extras,
            ) = track_completion(self._summary_df)
        return self._tracked_completion

    def _all_null(self, condition: Dict[int, int]) -> bool:
        return all([item == None for item in condition.values()])

    def _update(self, idx: int, condition: dict) -> None:
        if not self._all_null(condition):
            run_indices = [ix for ix in list(condition.values()) if not ix is None]
            train_result = self._summary_df.loc[run_indices]["train"].tolist()
            test_result = self._summary_df.loc[run_indices]["test"].tolist()
        else:
            train_result, test_result = [], []

        self._plot_data[f"cond_{idx}.train"] = train_result
        self._plot_data[f"cond_{idx}.test"] = test_result

    def _fetch_all_plot_data(self) -> None:
        for idx, condition in self.tracked_completion.items():
            self._update(idx=idx, condition=condition)

    def subset_plot_data_over_range(self, index_range: List[int]) -> Dict[str, List[float]]:
        plot_data_subset = {}
        for i in index_range:
            train = self._plot_data[f"cond_{i}.train"]
            test = self._plot_data[f"cond_{i}.test"]
            if len(train) == 0:
                train = [0]
            if len(test) == 0:
                test = [0]
            plot_data_subset.update({f"cond_{i}.train": train, f"cond_{i}.test": test})
        return plot_data_subset

    def __call__(
        self,
        indices: List[Tuple[int]] = [
            (0, 6),
            (6, 12),
            (12, 18),
            (18, 24),
            (24, 30),
            (30, 36),
            (36, 41),
        ],
    ):
        """ """

        plot_data = []
        for idx in indices:
            pl_subset = self.subset_plot_data_over_range(
                index_range=range(idx[0], idx[1])
            )
            plot_data.append(pl_subset)
        return plot_data

# -- function: ----------------------------------------------------------------
def aggregate_plot_data(
    summary_df: pd.DataFrame,
    indices: List[Tuple[int]] = [
        (0, 6),
        (6, 12),
        (12, 18),
        (18, 24),
        (24, 30),
        (30, 36),
        (36, 41),
    ],
) -> List[Dict[str, List[float]]]:
    aggr = PlotDataAggr(summary_df=summary_df)
    return aggr(indices=indices)
