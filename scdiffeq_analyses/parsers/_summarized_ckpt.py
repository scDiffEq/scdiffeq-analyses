# -- import packages: ---------------------------------------------------------
import ABCParse
import pandas as pd
import scdiffeq as sdq


# -- import local dependencies: ----------------------------------------------
from ._version_accuracy import VersionAccuracy


# -- set typing: --------------------------------------------------------------
from typing import List


# -- operational object class: ------------------------------------------------
class SummarizedCkpt(ABCParse.ABCParse):
    def __init__(self, project: sdq.io.Project):
        self.__parse__(locals())

    @property
    def versions(self) -> List[str]:
        return list(self._project._VERSION_PATHS.keys())

    def _get_version(self, version_name: str):
        return getattr(self._project, version_name)

    def __call__(self):

        BestResults = {}
        for version_name in self.versions:
            version = self._get_version(version_name=version_name)
            self.version_acc = VersionAccuracy(version=version)
            try:
                BestResults[version_name] = self.version_acc.best_test_from_train
            except:
                continue

        return pd.DataFrame(BestResults).T
