# -- import packages: ---------------------------------------------------------
import pandas as pd

# -- import local dependencies: -----------------------------------------------
from .. import types
from ._parse_fate_prediction_files import parse_fate_prediction_files

# -- cls: ---------------------------------------------------------------------
class RunAccuracy:
    def __init__(self, run: types.Run) -> None:
        self.files = parse_fate_prediction_files(run)

    def _get_ckpt_accuracy_frame(self, item: dict) -> pd.DataFrame:

        accuracy_csv_paths = list(item["fate-prediction-metrics"].glob("accuracy.csv"))
        ckpt_paths = list(item["model-ckpt"].glob("*"))

        assert len(ckpt_paths) == 1
        assert len(accuracy_csv_paths) == 1

        if item["epoch"] == "last":
            name = 2500
        elif item["epoch"] == "on_train_end":
            name = 2499
        else:
            name = int(item["epoch"])

        df = pd.read_csv(accuracy_csv_paths[0], index_col=0)
        df.columns = [name]
        return df

    def __call__(self) -> pd.DataFrame:
        df = pd.concat(
            [self._get_ckpt_accuracy_frame(item) for item in self.files], axis=1
        )
        return df[sorted(df.columns)]


# -- function: ----------------------------------------------------------------
def run_accuracy(run: types.Run) -> pd.DataFrame:
    run_accuracy = RunAccuracy(run=run)
    return run_accuracy()
