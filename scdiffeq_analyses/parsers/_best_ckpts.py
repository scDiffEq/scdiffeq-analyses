# -- import packages: ---------------------------------------------------------
import pandas as pd
import scdiffeq as sdq


# -- import local dependencies: -----------------------------------------------
from ._summarized_ckpt import SummarizedCkpt


# -- API-facing function: -----------------------------------------------------
def best_checkpoints(project: sdq.io.Project) -> pd.DataFrame:
    """Summarize the best checkpoint for each version of a project.

    Args:
        project (sdq.io.Project): An scdiffeq project instance.

    Returns:
        best_ckpts (pd.DataFrame): A DataFrame summarizing the best checkpoint for each version.
    """

    results = SummarizedCkpt(project=project)
    return results()
