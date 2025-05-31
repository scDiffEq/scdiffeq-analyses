# -- import packages: ---------------------------------------------------------
import pandas as pd
import pathlib
import wandb

# -- set type hints: ----------------------------------------------------------
from typing import Optional, Union

# -- function: ----------------------------------------------------------------
def download_run_history(
    run: wandb.apis.public.Run, fpath: Optional[Union[pathlib.Path, str]] = None
) -> pd.DataFrame:
    """Download the history of a run as a pandas DataFrame and save to .csv.

    Args:
        run: The run to download the history from.
        fpath: The path to save the history to.

    Returns:
        A pandas DataFrame containing the history of the run.
    """
    scan_history = run.scan_history()
    history = [item for item in scan_history]
    df = pd.DataFrame(history)
    if not fpath is None:
        df.to_csv(fpath)
    return df
