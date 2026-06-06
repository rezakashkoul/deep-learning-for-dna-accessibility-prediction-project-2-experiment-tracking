from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, random_split

SEED = 42

# One-hot encoding exactly as required in the guideline
# A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]
def one_hot_encode_dna(seq: str) -> torch.Tensor:
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }
    encoded = [mapping[base] for base in seq]
    return torch.tensor(encoded, dtype=torch.float32)


class DNADataset(Dataset):
    """
    Custom PyTorch Dataset that:
    - loads positive and negative text files
    - assigns binary labels
    - applies one-hot encoding on the fly
    """
    def __init__(self, positive_files, negative_files):
        self.samples = []

        # Load positive files and assign label 1
        for file_path in positive_files:
            file_path = Path(file_path)
            with open(file_path, "r") as f:
                for line in f:
                    seq = line.strip().upper()
                    if seq:
                        self.samples.append((seq, 1))

        # Load negative files and assign label 0
        for file_path in negative_files:
            file_path = Path(file_path)
            with open(file_path, "r") as f:
                for line in f:
                    seq = line.strip().upper()
                    if seq:
                        self.samples.append((seq, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, label = self.samples[idx]

        x = one_hot_encode_dna(sequence)   # shape: (174, 4)
        x = x.transpose(0, 1)              # shape: (4, 174) for Conv1D
        y = torch.tensor(label, dtype=torch.float32)

        return x, y


def _read_sequences(txt_path):
    sequences = []
    txt_path = Path(txt_path)
    with open(txt_path, "r") as f:
        for line in f:
            seq = line.strip().upper()
            if seq:
                sequences.append(seq)
    return sequences


def _discover_brain_experiments(brain_root):
    brain_root = Path(brain_root)
    experiment_dirs = sorted([p for p in brain_root.iterdir() if p.is_dir()])

    records = []
    for exp_dir in experiment_dirs:
        pos_files = sorted(exp_dir.glob("*_positive_*.txt"))
        neg_files = sorted(exp_dir.glob("*_negative_*.txt"))

        if len(pos_files) != 1 or len(neg_files) != 1:
            raise RuntimeError(f"Expected exactly one positive and one negative file in {exp_dir}")

        records.append({
            "experiment": exp_dir.name,
            "positive_file": pos_files[0],
            "negative_file": neg_files[0],
        })

    return records


def _discover_blood_files(blood_root):
    blood_root = Path(blood_root)
    experiment_dirs = sorted([p for p in blood_root.iterdir() if p.is_dir()])

    positive_files = []
    negative_files = []

    for exp_dir in experiment_dirs:
        pos_files = sorted(exp_dir.glob("*_positive_*.txt"))
        neg_files = sorted(exp_dir.glob("*_negative_*.txt"))

        if len(pos_files) != 1 or len(neg_files) != 1:
            raise RuntimeError(f"Expected exactly one positive and one negative file in {exp_dir}")

        positive_files.extend(pos_files)
        negative_files.extend(neg_files)

    return positive_files, negative_files


def build_balanced_brain_dataset(
    brain_root,
    per_experiment_per_class=5000,
    seed=SEED
):
    """
    Build the final balanced Brain dataset:
    - Use 5 brain experiments
    - Sample 5000 positive and 5000 negative from each experiment
    - Final size ~= 25,000 positive + 25,000 negative
    """
    rng = random.Random(seed)
    experiment_records = _discover_brain_experiments(brain_root)

    if len(experiment_records) != 5:
        raise RuntimeError(
            f"Expected exactly 5 brain experiments, but found {len(experiment_records)} in {brain_root}"
        )

    sampled_positive_paths = []
    sampled_negative_paths = []

    temp_root = Path("balanced_temp_brain_txt")
    temp_root.mkdir(parents=True, exist_ok=True)

    for record in experiment_records:
        exp_name = record["experiment"]

        pos_sequences = _read_sequences(record["positive_file"])
        neg_sequences = _read_sequences(record["negative_file"])

        if len(pos_sequences) < per_experiment_per_class:
            raise RuntimeError(
                f"{exp_name} positive has only {len(pos_sequences)} sequences; "
                f"need at least {per_experiment_per_class}"
            )

        if len(neg_sequences) < per_experiment_per_class:
            raise RuntimeError(
                f"{exp_name} negative has only {len(neg_sequences)} sequences; "
                f"need at least {per_experiment_per_class}"
            )

        pos_sample = rng.sample(pos_sequences, per_experiment_per_class)
        neg_sample = rng.sample(neg_sequences, per_experiment_per_class)

        pos_out = temp_root / f"{exp_name}_positive_balanced.txt"
        neg_out = temp_root / f"{exp_name}_negative_balanced.txt"

        with open(pos_out, "w") as f:
            for seq in pos_sample:
                f.write(seq + "\n")

        with open(neg_out, "w") as f:
            for seq in neg_sample:
                f.write(seq + "\n")

        sampled_positive_paths.append(pos_out)
        sampled_negative_paths.append(neg_out)

    dataset = DNADataset(sampled_positive_paths, sampled_negative_paths)
    return dataset


def build_brain_splits(
    brain_root,
    seed=SEED,
    per_experiment_per_class=5000
):
    """
    Create the final balanced Brain dataset and split it:
    Training 70%, Validation 15%, Testing 15%
    """
    full_dataset = build_balanced_brain_dataset(
        brain_root=brain_root,
        per_experiment_per_class=per_experiment_per_class,
        seed=seed
    )

    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def build_blood_dataset(blood_root):
    """
    Build Blood dataset directly from positive/negative text files.
    Used only for Experiment 5 external testing.
    """
    positive_files, negative_files = _discover_blood_files(blood_root)
    return DNADataset(positive_files, negative_files)

def _discover_heart_files(heart_root):
    heart_root = Path(heart_root)
    experiment_dirs = sorted([p for p in heart_root.iterdir() if p.is_dir()])

    positive_files = []
    negative_files = []

    for exp_dir in experiment_dirs:
        pos_files = sorted(exp_dir.glob("*_positive_*.txt"))
        neg_files = sorted(exp_dir.glob("*_negative_*.txt"))

        if len(pos_files) != 1 or len(neg_files) != 1:
            raise RuntimeError(f"Expected exactly one positive and one negative file in {exp_dir}")

        positive_files.extend(pos_files)
        negative_files.extend(neg_files)

    return positive_files, negative_files


def build_heart_dataset(heart_root):
    """
    Build Heart dataset directly from positive/negative text files.
    Used for frozen-backbone transfer learning in Project 2.
    """
    positive_files, negative_files = _discover_heart_files(heart_root)
    return DNADataset(positive_files, negative_files)
