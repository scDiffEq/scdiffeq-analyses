# -- import packages: ---------------------------------------------------------
import ABCParse
import logging
import pathlib
import wandb

# -- import local dependencies: -----------------------------------------------
from .. import types
from ._run import Run

# -- set type hints: ----------------------------------------------------------
from typing import List, Optional, Union

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- revised cls: -------------------------------------------------------------
class WandbClient(ABCParse.ABCParse):

    def __init__(
        self,
        project: str,
        entity: str = "scDiffEq",
        base_path: str = "scdiffeq_data/wandb",
    ) -> None:
        """"""

        self.__parse__(locals())
        self._init()

    @property
    def base_path(self) -> pathlib.Path:
        return pathlib.Path(self._base_path).joinpath(self._project)

    def _init(self) -> None:

        self._api = wandb.Api()
        self._runs = self._api.runs(f"{self._entity}/{self._project}")

    @property
    def runs(self) -> List[types.Run]:
        if not hasattr(self, "_sdq_runs"):
            self._sdq_runs = [Run(run=run, base_path=self.base_path) for run in self._runs]
        return self._sdq_runs


# -- cls: ---------------------------------------------------------------------
# class WandbClient:

#     def __init__(
#         self,
#         project: str,
#         entity: str = "scDiffEq",
#         base_path: Union[pathlib.Path, str] = pathlib.Path(
#             "scdiffeq_data/wandb_downloads"
#         ),
#         params: List[str] = [
#             "mu_hidden",
#             "sigma_hidden",
#             "velocity_ratio_params/enforce",
#             "velocity_ratio_params/target",
#         ],
#         n_target_files: int = 5,
#     ) -> None:

#         self._project = project
#         self._entity = entity
#         self._all_runs = []
#         self._complete_runs = []
#         self._failed_runs = []
#         self._base_path = pathlib.Path(base_path)
#         self._params = params
#         self._n_target_files = n_target_files
#         self._init()

#     def _init(self) -> None:

#         self._api = wandb.Api()
#         self._runs = self._api.runs(f"{self._entity}/{self._project}")

#     def get_runs(self, refresh: bool = True, limit: Optional[int] = None) -> None:
#         if refresh:
#             self._init()
#         self._all_runs = []
#         self._complete_runs = []
#         self._failed_runs = []
#         if limit:
#             iter_runs = self._runs[:limit]
#         else:
#             iter_runs = self._runs
#         for run in iter_runs:
#             try:
#                 sdq_run = Run(
#                     run=run,
#                     base_path=self._base_path,
#                     params=self._params,
#                     n_target_files=self._n_target_files,
#                 )
#                 sdq_run.download_artifacts()
#                 if sdq_run.is_complete:
#                     self._complete_runs.append(sdq_run)
#                 self._all_runs.append(sdq_run)
#             except Exception as e:
#                 logger.debug(
#                     f"Failed to process run '{run.name}' (ID: {run.id}). Error: {e}",
#                     exc_info=True
#                 )
#                 self._failed_runs.append(
#                     {"id": run.id, "name": run.name, "error": str(e)}
#                 )

#         n_all = len(self._all_runs)
#         n_complete = len(self._complete_runs)
#         n_failed = len(self._failed_runs)

#         logger.info(f"Found {n_all} runs ({n_complete} complete runs, {n_failed} runs)")

#     def _format_run_summary(self, min_benchmarked_ckpts: int = 5) -> list[str]:
#         """Formats a summary of runs, including only those with a minimum number of benchmarked checkpoints.

#         Args:
#             min_benchmarked_ckpts: The minimum number of benchmarked checkpoints a run must have to be included.

#         Returns:
#             A list of strings, where each string is a formatted line of the run summary.
#             The last string in the list is a total count of the runs included.
#         """
#         summary_lines = []
#         count = 0
#         for run in self._all_runs:
#             if len(run.benchmarked_ckpts) >= min_benchmarked_ckpts:
#                 summary_lines.append(
#                     f"{count:<3} | {run.name:<25} | [{len(run.benchmarked_ckpts):<2}] | {run.mu_hidden} {run.sigma_hidden}"
#                 )
#                 count += 1
#         summary_lines.append(f"//total: {count}")
#         return summary_lines

#     def print_run_summary(self) -> None:
#         print("\nFormatting run summary (default min_benchmarked_ckpts=5)...")
#         summary_lines_default = self._format_run_summary()
#         if summary_lines_default:
#             print("\n--- Run Summary (Default) ---")
#             for line in summary_lines_default:
#                 print(line)
#         else:
#             print("No runs met the default criteria for the summary.")

#         print(f"\nTotal runs processed: {len(self._all_runs)}")
#         print(f"Complete runs: {len(self._complete_runs)}")
#         print(f"Failed runs: {len(self._failed_runs)}")

#         if self._failed_runs:
#             print("Details of failed runs:")
#             for failed_run in self._failed_runs:
#                 print(
#                     f"  ID: {failed_run['id']}, Name: {failed_run['name']}, Error: {failed_run['error']}"
#                 )
