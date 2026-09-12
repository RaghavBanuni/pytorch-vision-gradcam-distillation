"""Command line interface: ``python -m visionlab.cli <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .augment import CUE_BREAKING, NO_AUGMENTATION, STANDARD, Augmentation
from .data import ImageConfig, build_dataset, describe
from .distill import DistillConfig, DistillTrainer, compression_report
from .evaluate import (
    evaluate_split,
    per_class_report,
    predict_logits,
    reliability_table,
    robustness_report,
    shortcut_gap,
    softmax_probabilities,
)
from .gradcam import localization_report
from .models import TinyCNN, build_model, count_parameters, freeze_backbone
from .train import TrainConfig, Trainer

AUGMENTATIONS: dict[str, Augmentation] = {
    "none": NO_AUGMENTATION,
    "standard": STANDARD,
    "cuebreak": CUE_BREAKING,
}


def _image_config(args: argparse.Namespace) -> ImageConfig:
    return ImageConfig(
        size=args.size,
        n_train=args.train_n,
        n_val=args.val_n,
        n_test=args.test_n,
        cue_strength=args.cue_strength,
        seed=args.seed,
    )


def _train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
    )


def _new_model(args: argparse.Namespace):
    kwargs = {} if args.model == "resnet18" else {"width": args.width}
    model = build_model(args.model, n_classes=4, pretrained=args.pretrained, **kwargs)
    if args.freeze:
        frozen = freeze_backbone(model)
        print(
            f"froze {frozen:,} of {count_parameters(model):,} parameters - "
            f"{count_parameters(model, trainable_only=True):,} still learning"
        )
    return model


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path}")


def _train(args: argparse.Namespace, datasets, augment: Augmentation, seed_offset: int = 0):
    model = _new_model(args)
    config = _train_config(args)
    if seed_offset:
        config = TrainConfig(**{**config.__dict__, "seed": config.seed + seed_offset})
    history = Trainer(config).fit(model, datasets["train"], datasets["val"], augment)
    return model, history


def cmd_data(args: argparse.Namespace) -> int:
    """Generate the dataset and show the shortcut that is planted in it."""
    datasets = build_dataset(_image_config(args))
    table = pd.DataFrame([describe(batch) for batch in datasets.values()])
    print("splits (cue_agreement is how often the background tint names the true class)\n")
    print(table.to_string(index=False), "\n")
    print(
        f"train and test share the same cue distribution ({args.cue_strength:.2f}), which is "
        "exactly why\nan i.i.d. test score cannot detect the shortcut - the shortcut is in the "
        "test set too.\nOn test_cue_inverted the tint names the wrong class, so a model that "
        "learned the tint\nscores far below chance while a model that learned the shape is "
        "unaffected."
    )
    if args.out:
        _write(table, Path(args.out) / "splits.csv")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train one model and report accuracy, calibration, robustness and localisation."""
    datasets = build_dataset(_image_config(args))
    augment = AUGMENTATIONS[args.augment]
    model, history = _train(args, datasets, augment)

    print(f"\ntrained {args.model} ({count_parameters(model):,} parameters)")
    print(f"augmentation: {args.augment}")
    print(history.frame().to_string(index=False))
    print(
        f"\nbest epoch {history.best_epoch} at val accuracy {history.best_val_accuracy:.4f}"
        + (" (stopped early)" if history.stopped_early else "")
    )

    report = robustness_report(model, datasets, device=args.device)
    print("\nthe same model on three test regimes")
    print(report.to_string(index=False))
    print(f"\nshortcut gap (test - inverted): {shortcut_gap(report):.4f}")

    logits = predict_logits(model, datasets["test"], args.device)
    probabilities = softmax_probabilities(logits)
    predictions = probabilities.argmax(axis=1)
    print("\nper class on the i.i.d. test split")
    print(per_class_report(predictions, datasets["test"].labels).to_string(index=False))
    print("\ncalibration on the i.i.d. test split")
    print(reliability_table(probabilities, datasets["test"].labels).to_string(index=False))

    summary, per_class = localization_report(
        model, datasets["test"], device=args.device
    )
    print("\nGrad-CAM localisation on the i.i.d. test split")
    print(json.dumps(summary, indent=2))
    print(per_class.to_string(index=False))
    print(
        "\npointing_game vs random_baseline is the whole point: a score at the baseline means "
        "the\nevidence for the prediction is not on the object."
    )

    if args.out:
        out = Path(args.out)
        _write(history.frame(), out / "history.csv")
        _write(report, out / "robustness.csv")
        _write(per_class, out / "localization_per_class.csv")
    if args.checkpoint:
        path = Path(args.checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": args.model, "width": args.width, "state_dict": model.state_dict()}, path)
        print(f"wrote {path}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Grad-CAM scoring on every test regime."""
    datasets = build_dataset(_image_config(args))
    if args.checkpoint and Path(args.checkpoint).exists():
        payload = torch.load(args.checkpoint, map_location=args.device)
        model = build_model(payload["model"], n_classes=4, width=payload.get("width", args.width))
        model.load_state_dict(payload["state_dict"])
        print(f"loaded {args.checkpoint}")
    else:
        model, _history = _train(args, datasets, AUGMENTATIONS[args.augment])

    rows = []
    for name in ("test", "test_cue_broken", "test_cue_inverted"):
        summary, _per_class = localization_report(model, datasets[name], device=args.device)
        rows.append(summary)
    table = pd.DataFrame(rows)
    print("\nlocalisation and accuracy across the three regimes")
    print(table.to_string(index=False), "\n")

    _summary, per_class = localization_report(model, datasets["test"], device=args.device)
    print("per class, i.i.d. test split")
    print(per_class.to_string(index=False))
    print(
        "\nGrad-CAM is a low-resolution, last-layer explanation; the pointing score says where "
        "the\nevidence sits, not why the network settled there."
    )
    if args.out:
        _write(table, Path(args.out) / "localization.csv")
    return 0


def cmd_robustness(args: argparse.Namespace) -> int:
    """Compare the interventions: plain training, cue-breaking augmentation, balanced data."""
    datasets = build_dataset(_image_config(args))
    runs: list[dict] = []

    for name, augment in (("standard", STANDARD), ("cue-breaking", CUE_BREAKING)):
        model, _history = _train(args, datasets, augment)
        report = robustness_report(model, datasets, device=args.device).set_index("split")
        summary, _per_class = localization_report(model, datasets["test"], device=args.device)
        runs.append(
            {
                "run": name,
                "train_cue_strength": args.cue_strength,
                "test": report.loc["test", "accuracy"],
                "cue_broken": report.loc["test_cue_broken", "accuracy"],
                "cue_inverted": report.loc["test_cue_inverted", "accuracy"],
                "shortcut_gap": round(
                    float(report.loc["test", "accuracy"] - report.loc["test_cue_inverted", "accuracy"]),
                    4,
                ),
                "pointing_game": summary["pointing_game"],
                "random_baseline": summary["random_baseline"],
            }
        )

    if args.balanced:
        balanced_config = ImageConfig(
            **{**_image_config(args).__dict__, "cue_strength": args.balanced_strength}
        )
        balanced = build_dataset(balanced_config)
        model, _history = _train(args, balanced, STANDARD)
        report = robustness_report(model, datasets, device=args.device).set_index("split")
        summary, _per_class = localization_report(model, datasets["test"], device=args.device)
        runs.append(
            {
                "run": "balanced data",
                "train_cue_strength": args.balanced_strength,
                "test": report.loc["test", "accuracy"],
                "cue_broken": report.loc["test_cue_broken", "accuracy"],
                "cue_inverted": report.loc["test_cue_inverted", "accuracy"],
                "shortcut_gap": round(
                    float(report.loc["test", "accuracy"] - report.loc["test_cue_inverted", "accuracy"]),
                    4,
                ),
                "pointing_game": summary["pointing_game"],
                "random_baseline": summary["random_baseline"],
            }
        )

    table = pd.DataFrame(runs)
    print("\ninterventions, all evaluated on the same three test sets")
    print(table.to_string(index=False), "\n")
    print(
        "Read the shortcut_gap column, not the test column. Every run looks similar on the "
        "i.i.d.\ntest set; they are not similar models. The balanced-data run is the honest fix "
        "and the one\nyou rarely get to make in production - which is why the augmentation row "
        "matters."
    )
    if args.out:
        _write(table, Path(args.out) / "interventions.csv")
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    """Teacher to student, against the same student trained on hard labels only."""
    datasets = build_dataset(_image_config(args))
    augment = AUGMENTATIONS[args.augment]

    teacher, teacher_history = _train(args, datasets, augment)
    print(f"\nteacher: {count_parameters(teacher):,} parameters, "
          f"best val {teacher_history.best_val_accuracy:.4f}")

    scratch = TinyCNN(n_classes=4, width=args.student_width)
    scratch_history = Trainer(_train_config(args)).fit(
        scratch, datasets["train"], datasets["val"], augment
    )
    print(f"student from scratch: best val {scratch_history.best_val_accuracy:.4f}")

    student = TinyCNN(n_classes=4, width=args.student_width)
    distilled_history = DistillTrainer(
        _train_config(args), DistillConfig(temperature=args.temperature, alpha=args.alpha)
    ).fit(student, teacher, datasets["train"], datasets["val"], augment)
    print(f"student distilled:   best val {distilled_history.best_val_accuracy:.4f}")

    rows = []
    for name, model in (("teacher", teacher), ("student_scratch", scratch), ("student_distilled", student)):
        for split in ("test", "test_cue_inverted"):
            result = evaluate_split(model, datasets[split], device=args.device)
            rows.append({"model": name, "parameters": count_parameters(model), **result})
    table = pd.DataFrame(rows)
    print("\naccuracy on the i.i.d. and inverted-cue test sets")
    print(table.to_string(index=False), "\n")

    print("compression")
    print(json.dumps(compression_report(teacher, student), indent=2))
    print(
        f"\nTemperature {args.temperature}, alpha {args.alpha}. The scratch student is the "
        "control:\nwithout it, any student result is unreadable, because a small model on an "
        "easy task may\nsimply not need a teacher."
    )
    if args.out:
        _write(table, Path(args.out) / "distillation.csv")
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--train-n", dest="train_n", type=int, default=2_400)
    parser.add_argument("--val-n", dest="val_n", type=int, default=600)
    parser.add_argument("--test-n", dest="test_n", type=int, default=600)
    parser.add_argument("--cue-strength", dest="cue_strength", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default="reports")


def _model_args(parser: argparse.ArgumentParser) -> None:
    _common(parser)
    parser.add_argument("--model", default="resnet_small", choices=["resnet_small", "tiny", "resnet18"])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--pretrained", action="store_true", help="download torchvision weights")
    parser.add_argument("--freeze", action="store_true", help="train the head only")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--augment", default="standard", choices=sorted(AUGMENTATIONS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionlab",
        description="Shortcut learning, scored Grad-CAM and distillation on procedural images.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="generate and describe the dataset")
    _common(data_parser)
    data_parser.set_defaults(func=cmd_data)

    train_parser = subparsers.add_parser("train", help="train and report every metric")
    _model_args(train_parser)
    train_parser.add_argument("--checkpoint", default="")
    train_parser.set_defaults(func=cmd_train)

    explain_parser = subparsers.add_parser("explain", help="Grad-CAM pointing game")
    _model_args(explain_parser)
    explain_parser.add_argument("--checkpoint", default="")
    explain_parser.set_defaults(func=cmd_explain)

    robust_parser = subparsers.add_parser("robustness", help="compare the interventions")
    _model_args(robust_parser)
    robust_parser.add_argument("--balanced", action="store_true", help="also retrain on balanced data")
    robust_parser.add_argument(
        "--balanced-strength", dest="balanced_strength", type=float, default=0.25
    )
    robust_parser.set_defaults(func=cmd_robustness)

    distill_parser = subparsers.add_parser("distill", help="teacher to student, with a control")
    _model_args(distill_parser)
    distill_parser.add_argument("--student-width", dest="student_width", type=int, default=8)
    distill_parser.add_argument("--temperature", type=float, default=4.0)
    distill_parser.add_argument("--alpha", type=float, default=0.7)
    distill_parser.set_defaults(func=cmd_distill)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
