# -- import packages: ---------------------------------------------------------
import sys
import os
import subprocess
import json
import time

# -- import local dependencies: -----------------------------------------------
from ._load_config import load_config

# -- operational cls: ---------------------------------------------------------
class ExperimentRunner:
    def __init__(self) -> None:
        ...

    def forward(self):
        ...

    def __call__(self):
        return self.forward()


# -- run function: ------------------------------------------------------------
def run_experiments():
    runner = ExperimentRunner()
    runner()


# def run_experiment(experiment_id: int, bucket_name: str) -> bool:

#     config = load_config()
#     experiment = None

#     # Find the experiment with the specified ID
#     for exp in config["experiments"]:
#         if exp["id"] == experiment_id:
#             experiment = exp
#             break

#     if experiment is None:
#         print(f"Experiment {experiment_id} not found in config")
#         return False

#     # Extract parameters for the experiment
#     mu_hidden = experiment["mu_hidden"]
#     sigma_hidden = experiment["sigma_hidden"]
#     velocity_ratio_target = experiment["velocity_ratio_target"]
#     velocity_ratio_enforce = experiment["velocity_ratio_enforce"]

#     # Extract common parameters
#     common = config["common"]
#     train_epochs = common["train_epochs"]
#     train_lr = common["train_lr"]
#     train_step_size = common["train_step_size"]
#     n_seeds = common["n_seeds"]
#     n_eval = common["n_eval"]
#     swa_lrs = common["swa_lrs"]
#     mu_dropout = common["mu_dropout"]
#     sigma_dropout = common["sigma_dropout"]
#     batch_size = common["batch_size"]
#     potential_type = common["potential_type"]
#     coef_g = common["coef_g"]
#     velocity_ratio_method = common["velocity_ratio_method"]

#     # Create output directory for this experiment
#     output_dir = f"experiment_{experiment_id}"
#     os.makedirs(output_dir, exist_ok=True)

#     # Build the command to run the experiment
#     cmd = [
#         "python",
#         "run.py",
#         f'--mu_hidden="{json.dumps(mu_hidden)}"',
#         f'--sigma_hidden="{json.dumps(sigma_hidden)}"',
#         f"--velocity_ratio_target={velocity_ratio_target}",
#         f"--velocity_ratio_enforce={velocity_ratio_enforce}",
#         f"--velocity_ratio_method={velocity_ratio_method}",
#         f"--train_epochs={train_epochs}",
#         f"--train_lr={train_lr}",
#         f"--train_step_size={train_step_size}",
#         f"--n_seeds={n_seeds}",
#         f"--n_eval={n_eval}",
#         f"--swa_lrs={swa_lrs}",
#         f"--mu_dropout={mu_dropout}",
#         f"--sigma_dropout={sigma_dropout}",
#         f"--batch_size={batch_size}",
#         f"--potential_type={potential_type}",
#         f"--coef_g={coef_g}",
#     ]

#     # Write command to log file
#     with open(f"{output_dir}/command.txt", "w") as f:
#         f.write(" ".join(cmd))

#     # Run the command and capture output
#     print(f"Starting experiment {experiment_id}...")
#     try:
#         with open(f"{output_dir}/output.log", "w") as f:
#             start_time = time.time()
#             process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
#             process.wait()
#             end_time = time.time()

#             # Record execution time
#             with open(f"{output_dir}/execution_time.txt", "w") as time_file:
#                 time_file.write(f"Execution time: {end_time - start_time} seconds")

#         # Upload results to GCS
#         upload_cmd = [
#             "gsutil",
#             "-m",
#             "cp",
#             "-r",
#             output_dir,
#             f"gs://{bucket_name}/results/",
#         ]
#         subprocess.run(upload_cmd, check=True)

#         print(f"Experiment {experiment_id} completed and results uploaded")
#         return True
#     except Exception as e:
#         print(f"Error running experiment {experiment_id}: {e}")
#         return False


# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python run_experiment.py EXPERIMENT_IDS BUCKET_NAME")
#         sys.exit(1)

#     experiment_ids = [int(x) for x in sys.argv[1].split(",")]
#     bucket_name = sys.argv[2]

#     for exp_id in experiment_ids:
#         run_experiment(exp_id, bucket_name)
