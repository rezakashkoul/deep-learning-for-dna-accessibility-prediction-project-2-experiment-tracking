import itertools
import subprocess
import sys

learning_rates = [0.001, 0.0001]
batch_sizes = [16, 64]
kernel_sizes = [5, 11]

epochs = 10

all_configs = list(itertools.product(learning_rates, batch_sizes, kernel_sizes))

print(f"Total runs: {len(all_configs)}")
print("=" * 80)

for idx, (lr, bs, ks) in enumerate(all_configs, start=1):
    run_name = f"grid_run_{idx}_lr_{lr}_bs_{bs}_ks_{ks}"

    cmd = [
        sys.executable,
        "train.py",
        "--kernel_size", str(ks),
        "--learning_rate", str(lr),
        "--batch_size", str(bs),
        "--epochs", str(epochs),
        "--run_name", run_name,
        "--results_root", "grid_results",
        "--log_dir", "runs_grid",
    ]

    print("=" * 80)
    print(f"Running {idx}/{len(all_configs)}")
    print("Command:", " ".join(cmd))
    print("=" * 80)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Run failed for: {run_name}")
        break

print("=" * 80)
print("Grid search finished.")
