
import pandas as pd
import pathlib


from typing import Dict


def larry_cytotrace() -> Dict[str, pd.DataFrame]:
    
    """ """
    
    _base_path = pathlib.Path(__file__).parents[2]
    
    obs_fpath = _base_path.joinpath("assets/larry.ct_obs_df.csv")
    var_fpath = _base_path.joinpath("assets/larry.ct_var_df.csv")

    obs_df = pd.read_csv(obs_fpath, index_col = 0)
    var_df = pd.read_csv(var_fpath, index_col = 0)

    obs_df.index = obs_df.index.astype(str)
    var_df.index = var_df.index.astype(str)

    return {
        "obs": obs_df,
        "var": var_df,
    }
