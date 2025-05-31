# -- import packages: ---------------------------------------------------------
import pandas as pd

# -- set type hints: ----------------------------------------------------------
from typing import Dict, Union


# -- function: ----------------------------------------------------------------
def get_best_ckpt(
    run_accuracy_df: pd.DataFrame,
    train_col: str = "unique_train.all_fates",
    test_col: str = "unique_test.all_fates",
) -> Dict[str, Union[float, str]]:
    df = run_accuracy_df.copy()
    df = (
        df.loc[[train_col, test_col]]
        .rename({train_col: "train", test_col: "test"}, axis=0)
        .T.reset_index()
        .rename({"index": "epoch"}, axis=1)
        .copy()
    )
    return (
        df.sort_values(["train", "epoch"], ascending=[False, False])
        .reset_index(drop=True)
        .loc[0]
        .to_dict()
    )
