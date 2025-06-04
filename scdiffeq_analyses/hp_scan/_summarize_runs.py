import pandas as pd

from typing import List

# -- import local dependencies: -----------------------------------------------
from .. import types
from ._run_accuracy import run_accuracy
from ._get_best_ckpt import get_best_ckpt

# -- function: ----------------------------------------------------------------
def summarize_runs(complete_runs: List[types.Run]) -> pd.DataFrame:
    best = []
    for run in complete_runs:
        acc_df = run_accuracy(run)
        best_ckpt = get_best_ckpt(acc_df)
        best_ckpt.update(
            {
                "mu_hidden": str(run.mu_hidden),
                "sigma_hidden": str(run.sigma_hidden),
                "E": str(run.velocity_ratio_params_enforce),
                "V": str(run.velocity_ratio_params_target),
                "seed": run.seed,
                "name": run.name,
            }
        )
        best.append(best_ckpt)
    df = pd.DataFrame(best).sort_values("test", ascending=False)
    df = df.rename({"epoch": "best_epoch"}, axis=1)
    df = df[
        [
            "name",
            "mu_hidden",
            "sigma_hidden",
            "V",
            "E",
            "seed",
            "best_epoch",
            "train",
            "test",
        ]
    ].copy()
    df = df.reset_index(drop=True).copy()
    return df
