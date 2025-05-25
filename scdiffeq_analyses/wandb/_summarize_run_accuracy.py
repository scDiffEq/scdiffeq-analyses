# -- import packages: ---------------------------------------------------------
import pandas as pd

# -- import local dependencies: -----------------------------------------------
from ._run import Run

# -- cls: ---------------------------------------------------------------------
class RunAccuracy:
    def __init__(self): ...

    def _convert_ckpt_name_to_epoch(self, ckpt_name):
        if "last_run_id" in ckpt_name:
            return 2500
        elif "on_train_end_run_id" in ckpt_name:
            return 2499
        else:
            return int(ckpt_name.split("epoch_")[1].split(".")[0])

    def _get_accuracy_csv(self, run, ckpt):
        p = list(run.dir.glob(f"*{ckpt}/accuracy.csv"))
        _df = pd.read_csv(p[0], index_col=0)
        _df.columns = [self._convert_ckpt_name_to_epoch(ckpt)]
        return _df

    def forward(self, run: Run) -> pd.DataFrame:
        df = pd.concat(
            [
                self._get_accuracy_csv(run, ckpt_name)
                for ckpt_name in run.benchmarked_ckpts
            ],
            axis=1,
        )
        return df[sorted(df.columns.tolist())].copy()

    def __call__(self, run: Run) -> pd.DataFrame:
        return self.forward(run=run)

# -- function: ----------------------------------------------------------------
def summarize_run_accuracy(run: Run) -> pd.DataFrame:
    run_accuracy = RunAccuracy()
    return run_accuracy(run=run)
