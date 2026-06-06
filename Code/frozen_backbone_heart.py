from pathlib import Path
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import build_heart_dataset
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

    return running_loss / total_samples, running_acc / total_samples


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

    return running_loss / total_samples, running_acc / total_samples


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
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "frozen_heart_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "frozen_heart_accuracy.png")
    plt.close()


def reinitialize_classifier(model):
    # Reinitialize fc1
    for module in model.fc1.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # Reinitialize fc2
    for module in model.fc2.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def main():
    parser = argparse.ArgumentParser(description="Frozen backbone transfer learning from Brain to Heart.")
    parser.add_argument("--heart_root", type=str, default="/users/kashkoul/project2/project_harmonized_174/heart")
    parser.add_argument("--best_model_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="runs_frozen_heart")
    parser.add_argument("--results_root", type=str, default="frozen_heart_results_reinit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"frozen_heart_reinit_{timestamp}"
    log_dir = Path(args.log_dir) / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print("TensorBoard log dir:", log_dir)

    outdir = Path(args.results_root)
    outdir.mkdir(parents=True, exist_ok=True)

    full_dataset = build_heart_dataset(args.heart_root)

    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    test_size = total_size - train_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = DNA1DCNN(kernel_size=11, seq_len=174, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(args.best_model_path, map_location=device))
    print("Loaded pretrained model from:", args.best_model_path)

    # Freeze backbone
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    for param in model.conv3.parameters():
        param.requires_grad = False

    # Reinitialize classifier exactly per guideline spirit
    reinitialize_classifier(model)
    print("Reinitialized fc1 and fc2 layers.")

    # Only classifier should be trainable
    for param in model.fc1.parameters():
        param.requires_grad = True
    for param in model.fc2.parameters():
        param.requires_grad = True

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )

    sample_x, _ = next(iter(train_loader))
    writer.add_graph(model, sample_x.to(device))

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_test_acc = -1.0
    best_frozen_model_path = outdir / "best_frozen_heart_model.pth"

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Test", test_loss, epoch + 1)
        writer.add_scalar("Accuracy/Test", test_acc, epoch + 1)

        print(
            f"FrozenHeartReinit Epoch [{epoch+1}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_frozen_model_path)
            print(f"New best frozen-heart model saved to: {best_frozen_model_path}")

    final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    cm, accuracy, precision, recall, f1 = confusion_metrics(model, test_loader, device)

    print()
    print(f"Frozen Heart Reinit Test Loss: {final_loss:.4f}")
    print(f"Frozen Heart Reinit Test Acc : {final_acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy  = {accuracy:.4f}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1-score  = {f1:.4f}")

    cm_fig = make_confusion_figure(cm, "Frozen Backbone Heart Reinitialized Classifier Confusion Matrix")
    writer.add_figure("Confusion_Matrix", cm_fig)
    plt.close(cm_fig)

    writer.add_hparams(
        {
            "transfer_learning": 1,
            "frozen_backbone": 1,
            "classifier_reinitialized": 1,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
        },
        {
            "hparam/final_accuracy": final_acc,
            "hparam/final_precision": precision,
            "hparam/final_recall": recall,
            "hparam/final_f1": f1,
        }
    )

    save_curves(history, outdir, "Frozen Heart Reinitialized")

    cm_save_fig = make_confusion_figure(cm, "Frozen Backbone Heart Reinitialized Classifier Confusion Matrix")
    cm_save_fig.savefig(outdir / "frozen_heart_cm.png")
    plt.close(cm_save_fig)

    with open(outdir / "metrics.txt", "w") as f:
        f.write("Frozen Backbone Heart Transfer Learning with Reinitialized Classifier\n")
        f.write(f"best_model_path_loaded={args.best_model_path}\n")
        f.write(f"best_frozen_model_path={best_frozen_model_path}\n")
        f.write("backbone_frozen=conv1,conv2,conv3\n")
        f.write("classifier_reinitialized=fc1,fc2\n")
        f.write("only_classifier_updated=True\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"learning_rate={args.learning_rate}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"test_loss={final_loss:.4f}\n")
        f.write(f"test_acc={final_acc:.4f}\n")
        f.write(f"accuracy={accuracy:.4f}\n")
        f.write(f"precision={precision:.4f}\n")
        f.write(f"recall={recall:.4f}\n")
        f.write(f"f1={f1:.4f}\n")
        f.write(f"confusion_matrix=\n{cm}\n")
        f.write(f"tensorboard_log_dir={log_dir}\n")

    writer.close()


if __name__ == "__main__":
    main()
