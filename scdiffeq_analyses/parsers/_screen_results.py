# -- import packages: ---------------------------------------------------------
import pathlib

# -- import local dependencies: -----------------------------------------------
from ._screen_result import ScreenResult

# -- set type hints: ----------------------------------------------------------
from typing import Iterator

# -- operational object cls: --------------------------------------------------
class ScreenResults:
    """Fate perturbation screen results"""
    def __init__(self, csv_dir: str) -> None:
        self._csv_dir = pathlib.Path(csv_dir)
        self._paths = list(self._csv_dir.glob("*"))
        self._get_data()

    def _get_versions(self) -> None:
        self._versions = []
        for path in self._paths:
            v = path.name.split(".")[0]
            if not v in self._versions:
                self._versions.append(v)
        self._versions = sorted(self._versions)

    def _get_data(self) -> None:
        self._get_versions()
        for version in self._versions:
            if not hasattr(self, version):
                base_path = self._csv_dir.joinpath(version)
                screen_result = ScreenResult(base_path)
                setattr(self, version, screen_result)

    def __repr__(self) -> str:
        return f"ScreenResults(csv_dir={self._csv_dir})"

    def __str__(self) -> str:
        return f"ScreenResults(csv_dir={self._csv_dir})"

    def __len__(self) -> int:
        return len(self._versions)
    
    def __getitem__(self, version: int) -> ScreenResult:
        return getattr(self, f"version_{version}")
    
    def __iter__(self) -> Iterator[ScreenResult]:
        for version in self._versions:
            yield getattr(self, version)
    
    def __contains__(self, version: str) -> bool:
        return version in self._versions
