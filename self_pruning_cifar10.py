from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PrunableLinear(nn.Module):
    """A linear layer whose weights are modulated by learnable sigmoid gates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initial_gate_score: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), initial_gate_score)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def gate_values(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def effective_weight(self, hard_threshold: float | None = None) -> torch.Tensor:
        gates = self.gate_values()
        if hard_threshold is not None:
            gates = gates * (gates >= hard_threshold).to(gates.dtype)
        return self.weight * gates

    def forward(self, x: torch.Tensor, hard_threshold: float | None = None) -> torch.Tensor:
        return F.linear(x, self.effective_weight(hard_threshold), self.bias)


class SelfPruningMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 32 * 32 * 3,
        hidden_dims: Sequence[int] = (512, 256, 128),
        num_classes: int = 10,
        dropout: float = 0.2,
        initial_gate_score: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(
                PrunableLinear(
                    previous_dim,
                    hidden_dim,
                    initial_gate_score=initial_gate_score,
                )
            )
            self.activations.append(nn.Sequential(nn.ReLU(), nn.Dropout(dropout)))
            previous_dim = hidden_dim

        self.output_layer = PrunableLinear(
            previous_dim,
            num_classes,
            initial_gate_score=initial_gate_score,
        )

    def prunable_layers(self) -> Iterable[PrunableLinear]:
        for layer in self.hidden_layers:
            yield layer
        yield self.output_layer

    def forward(self, x: torch.Tensor, hard_threshold: float | None = None) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        for layer, activation in zip(self.hidden_layers, self.activations):
            x = activation(layer(x, hard_threshold=hard_threshold))
        return self.output_layer(x, hard_threshold=hard_threshold)

    def sparsity_loss(self) -> torch.Tensor:
        penalties = [layer.gate_values().sum() for layer in self.prunable_layers()]
        return torch.stack(penalties).sum()

    def gate_values(self) -> torch.Tensor:
        return torch.cat([layer.gate_values().reshape(-1) for layer in self.prunable_layers()])


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    classification_loss: float
    sparsity_loss: float
    train_accuracy: float
    test_accuracy_soft: float
    test_accuracy_hard: float
    sparsity_level: float


@dataclass
class ExperimentResult:
    lambda_value: float
    epochs: int
    test_accuracy_soft: float
    test_accuracy_hard: float
    sparsity_level: float
    active_connections: int
    total_connections: int
    threshold: float
    gate_plot_path: str
    history: list[EpochMetrics]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def maybe_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    train_samples: int | None,
    test_samples: int | None,
    use_fake_data: bool,
) -> tuple[DataLoader, DataLoader, str]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    if use_fake_data:
        train_dataset = datasets.FakeData(
            size=train_samples or 2048,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
        )
        test_dataset = datasets.FakeData(
            size=test_samples or 512,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
        )
        dataset_name = "FakeData"
    else:
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        train_dataset = maybe_subset(train_dataset, train_samples)
        test_dataset = maybe_subset(test_dataset, test_samples)
        dataset_name = "CIFAR-10"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, dataset_name


def evaluate(
    model: SelfPruningMLP,
    loader: DataLoader,
    device: torch.device,
    hard_threshold: float | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs, hard_threshold=hard_threshold)
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    return correct / max(total, 1)


def compute_sparsity(model: SelfPruningMLP, threshold: float) -> tuple[float, int, int]:
    gates = model.gate_values()
    total_connections = gates.numel()
    active_connections = int((gates >= threshold).sum().item())
    sparsity_level = 100.0 * (1.0 - (active_connections / max(total_connections, 1)))
    return sparsity_level, active_connections, total_connections


def train_one_epoch(
    model: SelfPruningMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lambda_value: float,
) -> tuple[float, float, float, float]:
    model.train()
    running_total_loss = 0.0
    running_classification_loss = 0.0
    running_sparsity_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        classification_loss = criterion(logits, targets)
        sparsity_loss = model.sparsity_loss()
        total_loss = classification_loss + (lambda_value * sparsity_loss)
        total_loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_total_loss += total_loss.item() * batch_size
        running_classification_loss += classification_loss.item() * batch_size
        running_sparsity_loss += sparsity_loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    divisor = max(total, 1)
    return (
        running_total_loss / divisor,
        running_classification_loss / divisor,
        running_sparsity_loss / divisor,
        correct / divisor,
    )


def plot_gate_distribution(gates: torch.Tensor, output_path: Path, lambda_value: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.hist(gates.cpu().numpy(), bins=80, color="#1f77b4", edgecolor="white")
    plt.title(f"Gate Distribution for Best Model (lambda={lambda_value:.2e})")
    plt.xlabel("Gate value after sigmoid")
    plt.ylabel("Number of weights")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_summary_files(
    results: list[ExperimentResult],
    dataset_name: str,
    report_path: Path,
    csv_path: Path,
    json_path: Path,
    best_result: ExperimentResult,
    gate_threshold: float,
    args: argparse.Namespace,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lambda", "test_accuracy", "sparsity_level_percent", "active_connections", "total_connections"])
        for result in results:
            writer.writerow(
                [
                    f"{result.lambda_value:.2e}",
                    f"{result.test_accuracy_hard * 100:.2f}",
                    f"{result.sparsity_level:.2f}",
                    result.active_connections,
                    result.total_connections,
                ]
            )

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset_name,
                "gate_threshold": gate_threshold,
                "results": [asdict(result) for result in results],
            },
            handle,
            indent=2,
        )

    report_lines = [
        "# Self-Pruning Neural Network Report",
        "",
        "## Why an L1 penalty on sigmoid gates encourages sparsity",
        "",
        "Each gate contributes directly to the objective through `lambda * sum(sigmoid(gate_scores))`.",
        "That means every partially open connection pays a linear cost simply for staying active.",
        "During optimization, a weight keeps its gate open only if the classification gain offsets that cost.",
        "Connections that do not help enough are pushed toward smaller sigmoid values, which makes them easy to prune with a small threshold after training.",
        "",
        "## Experimental Setup",
        "",
        f"- Dataset: {dataset_name}",
        f"- Epochs per lambda: {args.epochs}",
        f"- Batch size: {args.batch_size}",
        f"- Hidden dimensions: {list(args.hidden_dims)}",
        f"- Base learning rate: {args.learning_rate}",
        f"- Gate learning-rate multiplier: {args.gate_lr_multiplier}",
        f"- Reported accuracy: hard-pruned test accuracy using a gate threshold of `{gate_threshold}`",
        f"- Best model selected by highest hard-pruned test accuracy: `lambda={best_result.lambda_value:.2e}`",
        "",
        "## Results",
        "",
        "| Lambda | Test Accuracy (%) | Sparsity Level (%) |",
        "| --- | ---: | ---: |",
    ]

    for result in results:
        report_lines.append(
            f"| `{result.lambda_value:.2e}` | {result.test_accuracy_hard * 100:.2f} | {result.sparsity_level:.2f} |"
        )

    if args.train_samples is not None or args.test_samples is not None:
        report_lines.extend(
            [
                "",
                "## Notes",
                "",
                f"This report was generated from a bounded run using `train_samples={args.train_samples}` and `test_samples={args.test_samples}`.",
                "Use the full CIFAR-10 splits and more epochs for a stronger sparsity-versus-accuracy study.",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Best Model Gate Distribution",
            "",
            f"![Gate distribution]({best_result.gate_plot_path.replace(chr(92), '/')})",
            "",
            "A successful pruning run should show many gates clustered near zero, while the remaining useful connections stay noticeably above the pruning threshold.",
        ]
    )

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run_experiment(
    lambda_value: float,
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    plot_filename: str,
) -> ExperimentResult:
    set_seed(args.seed)
    model = SelfPruningMLP(
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        initial_gate_score=args.initial_gate_score,
    ).to(device)
    gate_params: list[nn.Parameter] = []
    weight_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.endswith("gate_scores"):
            gate_params.append(parameter)
        else:
            weight_params.append(parameter)

    optimizer = torch.optim.Adam(
        [
            {
                "params": weight_params,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": gate_params,
                "lr": args.learning_rate * args.gate_lr_multiplier,
                "weight_decay": 0.0,
            },
        ]
    )
    criterion = nn.CrossEntropyLoss()
    history: list[EpochMetrics] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, classification_loss, sparsity_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            lambda_value=lambda_value,
        )
        soft_test_accuracy = evaluate(model, test_loader, device)
        hard_test_accuracy = evaluate(model, test_loader, device, hard_threshold=args.gate_threshold)
        sparsity_level, _, _ = compute_sparsity(model, args.gate_threshold)
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                classification_loss=classification_loss,
                sparsity_loss=sparsity_loss,
                train_accuracy=train_accuracy,
                test_accuracy_soft=soft_test_accuracy,
                test_accuracy_hard=hard_test_accuracy,
                sparsity_level=sparsity_level,
            )
        )
        print(
            f"[lambda={lambda_value:.2e}] epoch {epoch:02d}/{args.epochs} "
            f"train_acc={train_accuracy * 100:.2f}% "
            f"test_acc_soft={soft_test_accuracy * 100:.2f}% "
            f"test_acc_hard={hard_test_accuracy * 100:.2f}% "
            f"sparsity={sparsity_level:.2f}%"
        )

    final_sparsity, active_connections, total_connections = compute_sparsity(model, args.gate_threshold)
    gates = model.gate_values().detach().cpu()
    gate_plot_path = args.output_dir / plot_filename
    plot_gate_distribution(gates, gate_plot_path, lambda_value)

    return ExperimentResult(
        lambda_value=lambda_value,
        epochs=args.epochs,
        test_accuracy_soft=history[-1].test_accuracy_soft,
        test_accuracy_hard=history[-1].test_accuracy_hard,
        sparsity_level=final_sparsity,
        active_connections=active_connections,
        total_connections=total_connections,
        threshold=args.gate_threshold,
        gate_plot_path=str(gate_plot_path),
        history=history,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a self-pruning MLP on CIFAR-10.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--report-path", type=Path, default=Path("REPORT.md"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--gate-lr-multiplier",
        type=float,
        default=10.0,
        help="Multiplier applied to the learning rate of gate_scores parameters.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--initial-gate-score", type=float, default=0.0)
    parser.add_argument("--gate-threshold", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--use-fake-data", action="store_true")
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden layer widths for the MLP.",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1e-6, 5e-6, 1e-5],
        help="Sparsity regularization strengths to compare.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_loader, test_loader, dataset_name = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        use_fake_data=args.use_fake_data,
    )

    results: list[ExperimentResult] = []
    for lambda_value in args.lambdas:
        plot_filename = f"gate_distribution_lambda_{lambda_value:.0e}.png"
        results.append(
            run_experiment(
                lambda_value=lambda_value,
                args=args,
                device=device,
                train_loader=train_loader,
                test_loader=test_loader,
                plot_filename=plot_filename,
            )
        )

    best_result = max(results, key=lambda result: (result.test_accuracy_hard, result.sparsity_level))
    best_plot_path = Path(best_result.gate_plot_path)
    canonical_best_plot = args.output_dir / "best_model_gate_distribution.png"
    if best_plot_path != canonical_best_plot:
        canonical_best_plot.write_bytes(best_plot_path.read_bytes())
        best_result.gate_plot_path = str(canonical_best_plot)

    write_summary_files(
        results=results,
        dataset_name=dataset_name,
        report_path=args.report_path,
        csv_path=args.output_dir / "results_summary.csv",
        json_path=args.output_dir / "results_summary.json",
        best_result=best_result,
        gate_threshold=args.gate_threshold,
        args=args,
    )

    print("\nFinal summary:")
    for result in results:
        print(
            f"lambda={result.lambda_value:.2e} | "
            f"test_acc={result.test_accuracy_hard * 100:.2f}% | "
            f"sparsity={result.sparsity_level:.2f}%"
        )
    print(f"Best model report written to: {args.report_path}")


if __name__ == "__main__":
    main()
