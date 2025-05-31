# -- import packages: ---------------------------------------------------------
import ABCParse
import json
import logging
import pathlib
import pandas as pd
import wandb

# -- import local dependencies: -----------------------------------------------
from ._run_history import RunHistory
from ._run_memory import RunMemory

# -- set type hints: ----------------------------------------------------------
from typing import Any, Dict, List, Union

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- object cls: --------------------------------------------------------------
class Run(ABCParse.ABCParse):
    def __init__(
        self,
        run: wandb.apis.public.runs.Run,
        base_path: Union[pathlib.Path, str] = pathlib.Path(
            "scdiffeq_data/wandb_downloads"
        ),
    ) -> None:
        """ """

        self.__parse__(locals())

    @property
    def config(self) -> dict:
        if not hasattr(self, "_config"):
            config = json.loads(self._run.json_config)
            self._config = {key: val["value"] for key, val in config.items()}
        return self._config

    def get_configs(self, keys: List[str]) -> dict:
        return {key: self.config[key] for key in keys}

    @property
    def name(self) -> str:
        return self._run.name

    @property
    def id(self) -> str:
        return self._run.id

    @property
    def state(self) -> str:
        return self._run.state

    @property
    def save_name(self) -> str:
        if not hasattr(self, "_save_name"):
            name = self.name.replace("-", "_")
            self._save_name = f"{name}.{self.id}"
        return self._save_name

    @property
    def dir(self) -> pathlib.Path:
        if not hasattr(self, "_dir"):
            self._dir = self._base_path.joinpath(self.save_name)
            self._dir.mkdir(exist_ok=True, parents=True)
        return self._dir

    @property
    def history(self) -> pd.DataFrame:
        if not hasattr(self, "_history_df"):
            self._history = RunHistory(run=self)
            self._history_df = self._history.df
        return self._history_df

    @property
    def memory(self) -> pd.Series:
        if not hasattr(self, "_memory"):
            self._memory = RunMemory(history_df=self.history).max_requirements
        return self._memory

    @property
    def memory_requirements(self) -> dict:
        memory = {
            "id": self.id,
            "name": self.name,
        }
        memory.update(self.memory.to_dict())
        return memory

    def __repr__(self) -> str:
        try:
            return f"scDiffEq Run [wandb | name: {self.name} | run_id: {self.id}]"
        except Exception as e:
            return "scDiffEq Run [wandb]"


# class Run:
#     _DOWNLOAD_CALLED = False

#     def __init__(
#         self,
#         run: wandb.apis.public.runs.Run,
#         params: List[str] = [
#             "mu_hidden",
#             "sigma_hidden",
#             "velocity_ratio_params/enforce",
#             "velocity_ratio_params/target",
#         ],
#         base_path: Union[pathlib.Path, str] = pathlib.Path(
#             "scdiffeq_data/wandb_downloads"
#         ),
#         n_target_files: int = 5,
#     ) -> None:
#         """ """
#         self._run = run
#         self._params = params
#         self._n_target_files = n_target_files

#         self.state = self._run.state
#         self.name = self._run.name
#         self.id = self._run.id
#         if "seed" in self.config:
#             self.seed = self.config["seed"]["value"]
#         self._base_path = base_path

#         self._get_params_of_interest()

#     @property
#     def config(self) -> Dict[str, Any]:
#         if not hasattr(self, "_config"):
#             self._config = json.loads(self._run.json_config)
#         return self._config

#     def _get_params_of_interest(self) -> None:
#         for param in self._params:
#             try:
#                 if not "/" in param:
#                     setattr(self, param, self.config[param]["value"])
#                 else:
#                     k, v = param.split("/")
#                     setattr(self, param.replace("/", "_"), self.config[k]["value"][v])
#             except Exception as e:
#                 logger.debug(
#                     f"Could not set {param} to {self.config[param]['value']}, error: {e}"
#                 )

#     def download_artifacts(self, force: bool = False) -> None:

#         already_downloaded = 0
#         newly_downloaded = 0

#         self._DOWNLOAD_CALLED = True

#         self._artifact_dirs = []

#         for artifact in self._run.logged_artifacts():
#             DOWNLOADED = False
#             artifact_dir = self.dir.joinpath(artifact.name)
#             artifact_dir = artifact_dir.parent.joinpath(
#                 artifact_dir.name.replace("-", "_").replace("__", "_").replace(":", ".")
#             )
#             if not artifact_dir.exists():
#                 artifact_dir.mkdir(exist_ok=True, parents=True)
#             self._artifact_dirs.append(artifact_dir)
#             n_files = len(list(artifact_dir.glob("*")))
#             already_downloaded += n_files
#             if n_files == artifact.file_count:
#                 DOWNLOADED = True
#             if not DOWNLOADED or force:
#                 artifact.download(root=artifact_dir)
#                 newly_downloaded += artifact.file_count
#         logger.info(
#             f"{already_downloaded} files already downloaded. {newly_downloaded} files newly downloaded."
#         )

#     # def _get_history_path(self):
#     #     history_dirs = [path for path in self._artifact_dirs if "history" in str(path.name)]

#     #     if not history_dirs:
#     #         raise FileNotFoundError(
#     #             f"No artifact directory containing 'history' found for run {self.id} in {self.dir}. "
#     #             f"Available artifact_dirs: {[p.name for p in self._artifact_dirs]}"
#     #         )

#     #     # Use the first directory found that contains "history"
#     #     history_dir = history_dirs[0]
#     #     if len(history_dirs) > 1:
#     #         logger.warning(
#     #             f"Multiple artifact directories containing 'history' found for run {self.id}: "
#     #             f"{[p.name for p in history_dirs]}. Using '{history_dir.name}'."
#     #         )

#     #     parquet_files = sorted(list(history_dir.glob("*.parquet")))

#     #     if not parquet_files:
#     #         raise FileNotFoundError(
#     #             f"No .parquet file found in history artifact directory '{history_dir}' for run {self.id}."
#     #         )

#     #     history_file_path = parquet_files[0]
#     #     if len(parquet_files) > 1:
#     #         logger.warning(
#     #             f"Multiple .parquet files found in '{history_dir}': "
#     #             f"{[p.name for p in parquet_files]}. Using '{history_file_path.name}'."
#     #         )

#     #     return history_file_path


#     def _select_paths(self, key) -> list:
#         return [
#             path.name.split(f"{key}_")[1]
#             for path in self._artifact_dirs
#             if key in str(path.name)
#         ]

#     @property
#     def benchmarked_ckpts(self) -> List[str]:
#         """Checkpoints for which there are saved ckpts and corresponding metrics"""
#         if not hasattr(self, "_benchmarked_ckpts"):
#             if not self._DOWNLOAD_CALLED:
#                 self.download_artifacts()
#             self._ckpt_paths = self._select_paths(key="model_ckpt")
#             self._metrics_paths = self._select_paths(key="fate_prediction_metrics")
#             self._benchmarked_ckpts = list(
#                 set(self._metrics_paths) & set(self._ckpt_paths)
#             )
#         return self._benchmarked_ckpts

#     @property
#     def is_complete(self) -> bool:
#         return len(self.benchmarked_ckpts) >= self._n_target_files
