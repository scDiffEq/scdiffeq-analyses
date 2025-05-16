# -- import packages: --
import ABCParse
import numpy as np
import pandas as pd


# -- operational cls: ------
class CrossEntropy(ABCParse.ABCParse):
    def __init__(self, epsilon: float = 1e-12):
        self.__parse__(locals())
        
    def _augment_cols(self, F_obs, F_hat):
        for col in F_obs:
            if not col in F_hat:
                F_hat[col] = 0
        return F_hat
        
    def _norm(self, df: pd.DataFrame):
        return df.div(df.sum(axis=1), axis = 0)
    
    def _filter_zero_prediction_rows(self, F_obs: pd.DataFrame, F_hat: pd.DataFrame):
        mask = (F_hat.sum(1) > 0).values
#         print(mask.sum())
        F_obs = F_obs.loc[mask]
        F_hat = F_hat.loc[mask]
        return F_obs, F_hat, mask
    
    def __call__(self, F_obs, F_hat):
        
        true = F_obs.copy()
        pred = F_hat.copy()
        
            
        pred = self._augment_cols(F_obs = true, F_hat = pred)
        pred = pred[true.columns.tolist()]
        
#         true, pred, mask = self._filter_zero_prediction_rows(F_obs = true, F_hat = pred)    
        
        true['undiff'] = 0
        pred['undiff'] = 1 - pred.sum(1)
        
        
        pred = pred.clip(lower=self._epsilon)
        true = self._norm(df=true)
        pred = self._norm(df=pred)
        
        
        ce = -np.sum(true.values * np.log(pred.values + self._epsilon), axis=1)       
#         n_cells = mask.sum()
        return np.abs(ce).mean()

# -- api-facing function: ----    
def cross_entropy(F_obs: pd.DataFrame, F_hat: pd.DataFrame, epsilon: float = 1e-12):
    """"""
    calc = CrossEntropy(epsilon=epsilon)
    return calc(F_obs=F_obs, F_hat=F_hat)
