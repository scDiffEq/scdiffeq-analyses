# -- import packages: ---------------------------------------------------------
import pandas as pd

# -- import local dependencies: -----------------------------------------------
from ._get_conditions import get_conditions

# -- function: ----------------------------------------------------------------
# def get_conditions(condition):
#     defaults = {
#         "mu_hidden": [512, 512],
#         "sigma_hidden": [32, 32],
#         "velocity_ratio": {"enforce": 100, "target": 2.5},
#         "seed": -1,
#     }
#     updated_condition = {}
#     for key, val in defaults.items():
#         if key in condition:
#             val = condition[key]
#         updated_condition[key] = val
#     E = updated_condition["velocity_ratio"]["enforce"]
#     V = updated_condition["velocity_ratio"]["target"]
#     updated_condition.pop("velocity_ratio")
#     updated_condition["E"] = E
#     updated_condition["V"] = V
#     return updated_condition


def check_completion_tracker(tracked_completion_dict: dict):
    x = []
    for cond_i, done in tracked_completion_dict.items():
        for k, v in done.items():
            if not v is None:
                if not v in x:
                    x.append(v)
                else:
                    print(
                        f"Problem at: {cond_i}[{k}] - {v} has already been used elsewhere"
                    )


def track_completion(summary_df):
    # CONDITIONS = conditions.copy()
    CONDITIONS = get_conditions()
    available_index = list(summary_df.index).copy()
    used_index = []
    extras = []  # we want to avoid these
    N_SEEDS = 5
    use_count = 0
    tracked_completion_dict = {}
    condition_list_set = []
    for i in range(N_SEEDS):
        for en, original_condition in CONDITIONS.items():
            if not en in tracked_completion_dict:
                tracked_completion_dict[en] = {}

            # Create a copy of the condition to avoid modifying the original in CONDITIONS
            condition = original_condition.copy()
            condition.update({"seed": i})
            updated_condition = get_conditions(condition)
            condition_list_set.append(updated_condition)

            # Initialize completion_tracker for this seed to None
            tracked_completion_dict[en][i] = None

            complete_bool = pd.concat(
                [
                    summary_df[key].astype(str) == str(val)
                    for key, val in updated_condition.items()
                ],
                axis=1,
            )
            corresponding_indices = complete_bool[complete_bool.all(1)].index.tolist()

            for ix in corresponding_indices:
                if not ix in used_index:
                    tracked_completion_dict[en][i] = ix
                    used_index.append(ix)
                    use_count += 1
                    break  # Use the first available index and stop searching
    check_completion_tracker(tracked_completion_dict=tracked_completion_dict)
    for av_ix in available_index:
        if not av_ix in used_index:
            extras.append(av_ix)
    n_extras = len(extras)
    print(
        f"Found {n_extras} extras: {extras} (Matched: {use_count} / {len(available_index)} available)"
    )

    extras_df = summary_df.loc[extras]

    return tracked_completion_dict, condition_list_set, extras_df
