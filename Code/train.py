from pathlib import Path
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import build_brain_splits, build_blood_dataset
from model import DNA1DCNN


def binary_accuracy(preds, labels):
    predicted = (preds >= 0.5).float()
    correct = (predicted == labels).float().sum()
    return correct / labels.shape[0]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        batch_size = xb.size(0)
        acc = binary_accuracy(preds, yb)

        running_loss += loss.item() * batch_size
        running_acc += acc.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1)

        preds = model(xb)
        loss = criterion(preds, yb)

        batch_size = xb.size(0)
        acc = binary_accuracy(preds, yb)

        running_loss += loss.item() * batch_size
        running_acc += acc.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def confusion_metrics(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1)

        preds = model(xb)
        preds = (preds >= 0.5).float()

        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(yb.cpu().numpy().flatten())

    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)

    tp = int(((all_preds == 1) & (all_labels == 1)).sum())
    tn = int(((all_preds == 0) & (all_labels == 0)).sum())
    fp = int(((all_preds == 1) & (all_labels == 0)).sum())
    fn = int(((all_preds == 0) & (all_labels == 1)).sum())

    cm = np.array([[tn, fp], [fn, tp]])

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return cm, accuracy, precision, recall, f1


def make_confusion_figure(cm, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    return fig


def save_curves(history, outdir, title_prefix):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["eval_loss"], label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / f"{title_prefix.lower().replace(' ', '_')}_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["eval_acc"], label="Eval Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / f"{title_prefix.lower().replace(' ', '_')}_accuracy.png")
    plt.close()


def save_confusion_matrix(cm, outdir, title_prefix):
    fig = make_confusion_figure(cm, f"{title_prefix} Confusion Matrix")
    fig.savefig(outdir / f"{title_prefix.lower().replace(' ', '_')}_cm.png")
    plt.close(fig)


def get_experiment_config(experiment_id):
    if experiment_id == 1:
        return {"kernel_size": 5, "learning_rate": 0.001, "batch_size": 64, "title_prefix": "Experiment 1"}
    elif experiment_id == 2:
        return {"kernel_size": 11, "learning_rate": 0.001, "batch_size": 64, "title_prefix": "Experiment 2"}
    elif experiment_id == 3:
        return {"kernel_size": 5, "learning_rate": 0.0001, "batch_size": 64, "title_prefix": "Experiment 3"}
    elif experiment_id == 4:
        return {"kernel_size": 5, "learning_rate": 0.001, "batch_size": 16, "title_prefix": "Experiment 4"}
    elif experiment_id == 5:
        return {"kernel_size": 5, "learning_rate": 0.0001, "batch_size": 64, "title_prefix": "Experiment 5"}
    else:
        raise ValueError("Invalid experiment id")


def resolve_config(args):
    if args.experiment is not None:
        return get_experiment_config(args.experiment)

    if args.kernel_size is None or args.learning_rate is None or args.batch_size is None:
        raise ValueError(
            "When --experiment is not provided, you must provide "
            "--kernel_size, --learning_rate, and --batch_size."
        )

    custom_name = args.run_name if args.run_name is not None else "Custom Run"

    return {
        "kernel_size": args.kernel_size,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "title_prefix": custom_name
    }


def main():
    parser = argparse.ArgumentParser(description="Project 2 training with TensorBoard, checkpointing, and grid-search support.")
    parser.add_argument("--experiment", type=int, default=None, choices=[1, 2, 3, 4, 5])

    parser.add_argument("--kernel_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--brain_root", type=str, default="/users/kashkoul/project2/project_harmonized_174/brain")
    parser.add_argument("--blood_root", type=str, default="/users/kashkoul/project2/project_harmonized_174/blood")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--results_root", type=str, default="grid_results")
    parser.add_argument("--use_blood_eval", action="store_true")
    args = parser.parse_args()

    config = resolve_config(args)
    kernel_size = config["kernel_size"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    title_prefix = config["title_prefix"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    safe_name = title_prefix.lower().replace(" ", "_")
    outdir = Path(args.results_root) / safe_name
    outdir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{safe_name}_lr_{learning_rate}_bs_{batch_size}_ks_{kernel_size}_{timestamp}"
    log_dir = Path(args.log_dir) / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print("TensorBoard log dir:", log_dir)

    train_dataset, val_dataset, test_dataset = build_brain_splits(
        brain_root=args.brain_root,
        seed=args.seed,
        per_experiment_per_class=5000
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DNA1DCNN(kernel_size=kernel_size, seq_len=174, hidden_dim=256).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    sample_x, _ = next(iter(train_loader))
    writer.add_graph(model, sample_x.to(device))

    history = {
        "train_loss": [],
        "train_acc": [],
        "eval_loss": [],
        "eval_acc": []
    }

    best_eval_acc = -1.0
    best_model_path = outdir / "best_model.pth"

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        if args.use_blood_eval:
            eval_dataset = build_blood_dataset(args.blood_root)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
            eval_name = "Blood Test"
        else:
            eval_loader = test_loader
            eval_name = "Test"

        eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["eval_loss"].append(eval_loss)
        history["eval_acc"].append(eval_acc)

        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Test", eval_loss, epoch + 1)
        writer.add_scalar("Accuracy/Test", eval_acc, epoch + 1)

        print(
            f"{title_prefix} Epoch [{epoch+1}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"{eval_name} Loss: {eval_loss:.4f} | {eval_name} Acc: {eval_acc:.4f}"
        )

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to: {best_model_path}")

    if args.use_blood_eval:
        final_dataset = build_blood_dataset(args.blood_root)
        final_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=False)
        eval_name = "Blood Test"
    else:
        final_loader = test_loader
        eval_name = "Test"

    final_loss, final_acc = evaluate(model, final_loader, criterion, device)
    cm, accuracy, precision, recall, f1 = confusion_metrics(model, final_loader, device)

    print()
    print(f"{title_prefix} {eval_name} Loss: {final_loss:.4f}")
    print(f"{title_prefix} {eval_name} Acc : {final_acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy  = {accuracy:.4f}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1-score  = {f1:.4f}")

    cm_fig = make_confusion_figure(cm, f"{title_prefix} {eval_name} Confusion Matrix")
    writer.add_figure("Confusion_Matrix", cm_fig)
    plt.close(cm_fig)

    writer.add_hparams(
        {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "kernel_size": kernel_size,
            "epochs": args.epochs,
            "use_blood_eval": int(args.use_blood_eval),
        },
        {
            "hparam/final_accuracy": final_acc,
            "hparam/final_precision": precision,
            "hparam/final_recall": recall,
            "hparam/final_f1": f1,
        }
    )

    save_curves(history, outdir, title_prefix)
    save_confusion_matrix(cm, outdir, title_prefix)

    with open(outdir / "metrics.txt", "w") as f:
        f.write(f"{title_prefix}\n")
        f.write(f"kernel_size={kernel_size}\n")
        f.write(f"learning_rate={learning_rate}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"use_blood_eval={args.use_blood_eval}\n")
        f.write(f"best_model_path={best_model_path}\n")
        f.write(f"{eval_name.lower().replace(' ', '_')}_loss={final_loss:.4f}\n")
        f.write(f"{eval_name.lower().replace(' ', '_')}_acc={final_acc:.4f}\n")
        f.write(f"accuracy={accuracy:.4f}\n")
        f.write(f"precision={precision:.4f}\n")
        f.write(f"recall={recall:.4f}\n")
        f.write(f"f1={f1:.4f}\n")
        f.write(f"confusion_matrix=\n{cm}\n")
        f.write(f"tensorboard_log_dir={log_dir}\n")

    writer.close()


if __name__ == "__main__":
    main()
