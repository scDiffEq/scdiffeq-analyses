# -- import packages: ---------------------------------------------------------
import ABCParse
import pandas as pd
import pathlib
import scdiffeq as sdq


# -- set typing: --------------------------------------------------------------
from typing import List


# -- operational object class: ------------------------------------------------
class VersionAccuracy(ABCParse.ABCParse):
    def __init__(
        self,
        version: sdq.io.Version,
        train_col: str = "unique_train.all_fates",
        test_col: str = "unique_test.all_fates",
        *args,
        **kwargs,
    ):
        self.__parse__(locals())

    @property
    def csv_paths(self) -> List[pathlib.Path]:
        return list(self._version._PATH.glob("fate_prediction_metrics/*/accuracy.csv"))

    def _read_frame(self, path: pathlib.Path):

        name = path.parent.name
        return (
            pd.read_csv(path, index_col=0)
            .loc[[self._train_col, self._test_col]]
            .rename({self._train_col: "train", self._test_col: "test"}, axis=0)
            .rename({"accuracy": name}, axis=1)
        )

    def _format_ckpt_name(self, name):
        return name.replace("=", "_").split(".ckpt")[0].replace("-", ".")

    @property
    def _saved_ckpt_fpaths(self) -> List[pathlib.Path]:
        if not hasattr(self, "_cached_saved_ckpt_fpaths"):
            raw_paths: List[pathlib.Path] = []
            if self._version and hasattr(self._version, 'ckpts') and self._version.ckpts:
                items_to_iterate = []
                if isinstance(self._version.ckpts, dict):
                    items_to_iterate = self._version.ckpts.values()
                elif isinstance(self._version.ckpts, list):
                    items_to_iterate = self._version.ckpts
                
                for ckpt_obj in items_to_iterate:
                    if hasattr(ckpt_obj, 'path') and isinstance(getattr(ckpt_obj, 'path'), pathlib.Path):
                        raw_paths.append(getattr(ckpt_obj, 'path'))
            self._cached_saved_ckpt_fpaths = raw_paths
        return self._cached_saved_ckpt_fpaths

    @property
    def _saved_ckpts(self):
        """Not all (e.g., `on_train_epoch_end`) ckpts get saved. Use this to filter accordingly"""
        if not hasattr(self, "_cached_saved_ckpts_names"):
            self._cached_saved_ckpts_names = [self._format_ckpt_name(p.name) for p in self._saved_ckpt_fpaths]
        return self._cached_saved_ckpts_names

    @property
    def _ckpt_name_to_path_mapping(self) -> dict[str, pathlib.Path]:
        if not hasattr(self, "_internal_ckpt_name_to_path_map_cache"):
            mapping = {}
            for p in self._saved_ckpt_fpaths:  # These are Path objects from _saved_ckpt_fpaths
                formatted_name = self._format_ckpt_name(p.name)
                path_to_store = p  # Default to the original path

                if p.name == "last.ckpt": # Only apply special logic for "last.ckpt"
                    if p.is_symlink():
                        try:
                            # Attempt to resolve the symlink strictly (target must exist)
                            resolved_path = p.resolve(strict=True)
                            path_to_store = resolved_path
                        except FileNotFoundError:
                            # Symlink is broken or target doesn't exist.
                            # Try to read what it points to and construct the intended path on the CURRENT machine.
                            try:
                                target_link_str = p.readlink() # Gets the path the symlink points to as a string
                                target_path_obj = pathlib.Path(target_link_str) # Convert to Path object

                                if target_path_obj.is_absolute():
                                    # If the link target was an absolute path (from another machine),
                                    # we only care about its filename and assume it should be
                                    # relative to the symlink's location on *this* machine.
                                    actual_filename = target_path_obj.name
                                    intended_path = (p.parent / actual_filename).resolve(strict=False)
                                else:
                                    # If the link target was relative, interpret it relative to the symlink's parent.
                                    intended_path = (p.parent / target_path_obj).resolve(strict=False)
                                
                                path_to_store = intended_path
                                # print(f"Warning: Symlink {p} is broken. Using re-interpreted target: {intended_path}")
                            except Exception as e_readlink_reconstruct:
                                # Failed to readlink or reconstruct path (e.g., permissions, or not a symlink after all)
                                # print(f"Warning: Could not reconstruct path for broken symlink {p}: {e_readlink_reconstruct}. Using original path {p}.")
                                path_to_store = p # Fallback to original symlink path
                        except Exception as e_resolve:
                            # Other errors during p.resolve(strict=True) (e.g. permission errors not FileNotFoundError)
                            # print(f"Warning: Error resolving symlink {p} strictly: {e_resolve}. Using original path {p}.")
                            path_to_store = p # Fallback to original symlink path
                    # else: # It's a regular file named "last.ckpt", not a symlink.
                        # print(f"Info: 'last.ckpt' at {p} is a regular file, not a symlink. Using its direct path.")
                        # path_to_store remains p, which is correct for a non-symlink.
                
                mapping[formatted_name] = path_to_store
            self._internal_ckpt_name_to_path_map_cache = mapping
        return self._internal_ckpt_name_to_path_map_cache

    def _get_ckpt_epoch_values(self, df_with_paths: pd.DataFrame) -> List[int]:
        epochs = []
        for original_idx_name, row in df_with_paths.iterrows():
            # original_idx_name is from df.index, e.g., "last", "epoch_123.v1"
            # Split off potential versioning like ".v1" to get the base epoch name part
            current_epoch_str_part = original_idx_name.split('.')[0]  # "epoch_123.v1" -> "epoch_123"; "last" -> "last"

            if current_epoch_str_part == "last":
                ckpt_path_obj = row["ckpt_path"]  # This is a pathlib.Path, potentially resolved
                actual_file_name = ckpt_path_obj.name  # e.g., "epoch_2499.ckpt" or "last.ckpt"
                
                # Use _format_ckpt_name to standardize the name before parsing epoch
                # e.g., "epoch_2499.ckpt" -> "epoch_2499"
                formatted_name_for_epoch_parsing = self._format_ckpt_name(actual_file_name)

                if formatted_name_for_epoch_parsing.startswith("epoch_"):
                    try:
                        # Extract epoch number, e.g., from "epoch_2499"
                        epoch_val = int(formatted_name_for_epoch_parsing.split("epoch_")[1])
                        epochs.append(epoch_val)
                    except (ValueError, IndexError):
                        # Fallback if parsing from resolved name (e.g. "epoch_XYZ") fails
                        epochs.append(2500) 
                else:
                    # Fallback if resolved name isn't "epoch_XXX" (e.g., if it's still "last" or something else)
                    epochs.append(2500)
            elif current_epoch_str_part.startswith("epoch_"):
                try:
                    # Extract epoch number from names like "epoch_123"
                    epoch_val = int(current_epoch_str_part.split("epoch_")[1])
                    epochs.append(epoch_val)
                except (ValueError, IndexError):
                    # Fallback for regular epoch parsing failure
                    # print(f"Warning: Could not parse epoch from '{current_epoch_str_part}'")
                    epochs.append(-1)  # Placeholder for unparsable epoch
            else:
                # Handle other non-"epoch_" or "last" names if they exist
                # print(f"Warning: Unrecognized epoch format: '{current_epoch_str_part}'")
                epochs.append(-1)  # Placeholder for unknown formats
        return epochs

    @property
    def df(self):
        if not hasattr(self, "_df"):
            df_parts = [self._read_frame(path) for path in self.csv_paths]
            
            if not df_parts:
                self._df = pd.DataFrame(columns=["train", "test", "ckpt_path", "epoch"])
                return self._df

            df = pd.concat(
                df_parts, axis=1
            ).T
            
            df = df.loc[df.index.isin(self._saved_ckpts)]
            
            df["ckpt_path"] = df.index.map(self._ckpt_name_to_path_mapping)
            
            df.dropna(subset=["ckpt_path"], inplace=True)
            
            df["epoch"] = self._get_ckpt_epoch_values(df)
            self._df = df.sort_values(["train", "epoch"], ascending=[False, False])
        return self._df

    @property
    def best_training_ckpt(self):
        return self.df.index[0]

    @property
    def best_test_from_train(self) -> pd.Series:
        return self.df.loc[self.best_training_ckpt]
