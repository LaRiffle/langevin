import argparse
import datetime
import itertools
import shutil
import subprocess
from pathlib import Path, PurePath
from time import sleep

import yaml

# All the boolean arguments with "store_true" and default=False
# We can move these lists to the Yaml file to make the script more reusable
BOOLEAN_FLAGS = [
    "exclusive",
    "save-model",
    "disable-dp",
    "secure-rng",
    "clip-per-layer",
]
POSITIONAL_ARGUMENTS = ["data"]


def load_yaml(path):
    with open(path, "r") as f:
        c = yaml.load(f, Loader=yaml.loader.SafeLoader)
    return c


def expand_configs(configs):
    if isinstance(configs, list):
        return configs
    if isinstance(configs, dict):
        d = {}
        for (key, value) in configs.items():
            if isinstance(value, list):
                d[key] = value
            # TODO: if necessary, add support for nested dictionaries
            else:
                d[key] = [value]

        # Compute the nested cartesian product
        return [dict(zip(d, x)) for x in itertools.product(*d.values())]


def run_experiment(config_paths, dry_run, n_runs, logs_root):
    for yaml_path in config_paths:
        for _ in range(n_runs):
            generate_run_configs(yaml_path, dry_run, logs_root)
            sleep(1)  # Ensure that experiments have different names


def generate_run_configs(yaml_path, dry_run, logs_root):
    experiment = load_yaml(yaml_path)
    experiment_uid = (
        PurePath(yaml_path).stem + "-" + datetime.datetime.now().strftime("%m%d-%H%M%S")
    )

    print(f"Running experiment {experiment_uid} with configuration: \n {experiment}")

    exp_logs = logs_root.joinpath(experiment_uid)
    exp_logs.mkdir(exist_ok=True, parents=True)

    # Save a copy of the experiment parameters
    shutil.copy(yaml_path, exp_logs.joinpath(PurePath(yaml_path).name))

    environment = experiment["environment"]
    configs = expand_configs(experiment["arguments"])
    # Generate the batch scripts, and optionally run them
    for config_id, config in enumerate(configs):
        # Each job has its own output dir
        config_logs = exp_logs.joinpath(f"{config_id}")
        config_logs.mkdir(exist_ok=True, parents=True)

        # Save a copy of the Python script
        python_path = config["python_file"]
        shutil.copy(python_path, config_logs.joinpath(PurePath(python_path).name))

        # Update the logs dir for Slurm and the Python metrics
        environment["output"] = str(config_logs.joinpath("%x-%j.out"))
        config["metrics"] = str(config_logs.joinpath(f"metrics-{experiment_uid}-{config_id}.yaml"))
        with open(config_logs.joinpath(f"config-{experiment_uid}-{config_id}.yaml"), "w") as f:
            yaml.dump(config, f)

        print(f"Running configuration {config_id} with parameters: \n {config}")

        # Write and run the batch script
        script_path = config_logs.joinpath(f"batch-{experiment_uid}-{config_id}.sh")
        write_batch_script(environment.copy(), config, script_path)
        if not dry_run:
            try:
                subprocess.run(
                    f"{'module load cgpu; ' if 'modules' in environment else ''}sbatch {script_path}",
                    shell=True,
                )
            except Exception as e:
                print(f"This configuration failed. \n {e}")


def write_batch_script(environment, arguments, output_path):
    with open(output_path, "w") as f:
        f.write("#!/bin/bash\n")
        modules = environment.pop("modules", None)
        precommands = environment.pop("precommands")

        # Write the #SBATCH options
        for key, value in environment.items():
            if key in BOOLEAN_FLAGS:
                if value:
                    f.write(f"#SBATCH --{key}\n")
            else:
                f.write(f"#SBATCH --{key}={value}\n")

        # Load the modules.
        if modules:
            f.write("module purge\n")
            for module in modules:
                f.write(f"module load {module}\n")

        # Run arbitrary precommands
        for precommand in precommands:
            f.write(f"{precommand}\n")

        # Prepare the command line arguments
        python_executable_path = arguments.pop("python_executable", "python")  # optional
        python_file_path = arguments.pop("python_file")
        command = ["srun", python_executable_path, python_file_path]
        for key, value in arguments.items():
            if key in POSITIONAL_ARGUMENTS:
                command.append(f"{value}")
            else:
                if not key in BOOLEAN_FLAGS:
                    command.append(f"--{key}")
                    command.append(f"{value}")
                else:
                    if value:
                        command.append(f"--{key}")

        # Run the command
        f.write(" ".join(command))
        f.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_paths",
        type=str,
        nargs="+",
        help="List of space-separated paths to Yaml config files.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Prepare the batch scripts but don't execute anything.",
    )
    parser.add_argument(
        "--logs-root",
        type=str,
        default="/global/cfs/cdirs/m1759/exappai/logs",
        help="Base path to store the logs. Default: /global/cfs/cdirs/m1759/exappai/logs",
    )
    parser.add_argument(
        "--logs-prefix",
        type=str,
        default="",
        help="Custom prefix to group the logs of multiple files or multiple iterations. Default: ''.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of repetitions for each configuration. Default: 1.",
    )
    args = parser.parse_args()

    logs_root = Path(args.logs_root).joinpath(args.logs_prefix)

    run_experiment(args.config_paths, args.dry_run, args.n_runs, logs_root)
