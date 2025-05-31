# -- import packages: ---------------------------------------------------------
import collections
import re

# -- import local dependencies: -----------------------------------------------
from .. import types

def parse_fate_prediction_files(run: types.Run):

    file_list = [fpath.name for fpath in list(run.dir.glob("*"))]

    # Updated patterns to handle both underscore and hyphen cases in 'run_id' part
    ckpt_pattern = re.compile(
        r"model-ckpt-seed_(\d+)-(?:epoch_(\d+)\.step_(\d+)|([a-z_]+))-run_id[-_]([a-f0-9]+)_v0"
    )
    metrics_pattern = re.compile(
        r"fate-prediction-metrics-seed_(\d+)-(?:epoch_(\d+)\.step_(\d+)|([a-z_]+))-run_id[-_]([a-f0-9]+)_v0"
    )

    combined = collections.defaultdict(dict)

    for fname in file_list:
        fname = str(fname)

        m_ckpt = ckpt_pattern.fullmatch(fname)
        if m_ckpt:
            seed, epoch, step, special, run_id = m_ckpt.groups()
            epoch = epoch if epoch else special
            step = step if step else special
            key = (run_id, int(seed), epoch)
            combined[key]["run_id"] = run_id
            combined[key]["seed"] = int(seed)
            combined[key]["epoch"] = epoch
            combined[key]["step"] = step
            combined[key]["model-ckpt"] = run.dir.joinpath(fname)
            continue

        m_metrics = metrics_pattern.fullmatch(fname)
        if m_metrics:
            seed, epoch, step, special, run_id = m_metrics.groups()
            epoch = epoch if epoch else special
            step = step if step else special
            key = (run_id, int(seed), epoch)
            combined[key]["run_id"] = run_id
            combined[key]["seed"] = int(seed)
            combined[key]["epoch"] = epoch
            combined[key]["step"] = step
            combined[key]["fate-prediction-metrics"] = run.dir.joinpath(fname)
            continue

    # Filter complete pairs
    results = []
    for item in combined.values():
        if "model-ckpt" in item and "fate-prediction-metrics" in item:
            results.append(item)

    return results
