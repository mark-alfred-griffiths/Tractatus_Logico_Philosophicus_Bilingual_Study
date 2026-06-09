from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def lambda_tag(value: float) -> str:
    return f"align{int(round(value * 100)):03d}"


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def ensure_layout(study_dir: Path) -> None:
    for name in ["checkpoints", "logs", "metrics", "latents", "summaries"]:
        path = study_dir / name
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch(exist_ok=True)


def write_manifest(study_dir: Path, alignment_lambda: float, seeds: list[int], args: argparse.Namespace) -> None:
    manifest = {
        "study": study_dir.name,
        "description": "Bilingual alignment-strength seed sweep.",
        "lambda_language_alignment": alignment_lambda,
        "seeds": seeds,
        "data": str(args.data),
        "languages": ["en", "de"],
        "model": {
            "split_latent": True,
            "text_latent_dim": args.text_latent_dim,
            "structure_latent_dim": args.structure_latent_dim,
            "language_embedding_dim": args.language_embedding_dim,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "beta": args.beta,
            "beta_text": args.beta_text,
            "beta_structure": args.beta_structure,
            "lambda_parent": args.lambda_parent,
            "lambda_depth": args.lambda_depth,
            "lambda_next": args.lambda_next,
            "lambda_child": args.lambda_child,
            "lambda_language_alignment": alignment_lambda,
            "lr": args.lr,
            "device": args.device,
        },
        "paths": {
            "checkpoints": "checkpoints",
            "logs": "logs",
            "metrics": "metrics",
            "latents": "latents",
            "summaries": "summaries",
        },
    }
    (study_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_command(command: list[str], log_path: Path | None = None, stdout_path: Path | None = None, dry_run: bool = False) -> None:
    printable = " ".join(command)
    if dry_run:
        print(printable)
        return
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            subprocess.run(command, check=True, text=True, stdout=handle, stderr=subprocess.STDOUT)
        return
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as handle:
            subprocess.run(command, check=True, text=True, stdout=handle)
        return
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bilingual alignment-strength sweeps across multiple seeds.")
    parser.add_argument("--out-root", type=Path, default=Path("runs/seed_sweeps/bilingual_alignment_lambda_sweep"))
    parser.add_argument("--data", type=Path, default=Path("tractatus_structure_latents/data/tractatus_bilingual.json"))
    parser.add_argument("--lambdas", default="0.00,0.03,0.10,0.30,1.00", help="Comma-separated lambda_language_alignment values.")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated integer seeds.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-latent-dim", type=int, default=24)
    parser.add_argument("--structure-latent-dim", type=int, default=8)
    parser.add_argument("--language-embedding-dim", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--beta-text", type=float, default=0.01)
    parser.add_argument("--beta-structure", type=float, default=0.05)
    parser.add_argument("--lambda-parent", type=float, default=0.2)
    parser.add_argument("--lambda-depth", type=float, default=0.1)
    parser.add_argument("--lambda-next", type=float, default=0.2)
    parser.add_argument("--lambda-child", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-existing", action="store_true", help="Skip train/eval when both checkpoint and metrics file already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    alignment_values = parse_floats(args.lambdas)
    seeds = parse_ints(args.seeds)
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "BILINGUAL_ALIGNMENT_LAMBDA_SWEEP.md").write_text(
        "# Bilingual Alignment Lambda Seed Sweep\n\n"
        "Each `alignXYZ/` folder contains one alignment strength repeated across seeds. "
        "Use `tractatus_structure_latents.evaluation.plot_bilingual_alignment_sweep --seed-sweep-dir` "
        "to aggregate mean/std figures.\n",
        encoding="utf-8",
    )

    for alignment_lambda in alignment_values:
        tag = lambda_tag(alignment_lambda)
        study_dir = args.out_root / tag
        ensure_layout(study_dir)
        write_manifest(study_dir, alignment_lambda, seeds, args)
        for seed in seeds:
            seed_name = f"seed{seed:03d}"
            checkpoint = study_dir / "checkpoints" / f"{seed_name}.pt"
            metric_path = study_dir / "metrics" / f"{seed_name}.metrics.json"
            latent_path = study_dir / "latents" / f"{seed_name}_structure.pt"
            log_path = study_dir / "logs" / f"{seed_name}.train.log"
            if args.skip_existing and checkpoint.exists() and metric_path.exists():
                print(f"skipping {tag}/{seed_name}; checkpoint and metrics already exist")
                continue

            train_command = [
                "python3",
                "-m",
                "tractatus_structure_latents.training.train_vae",
                "--data",
                str(args.data),
                "--split-latent",
                "--text-latent-dim",
                str(args.text_latent_dim),
                "--structure-latent-dim",
                str(args.structure_latent_dim),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--beta",
                str(args.beta),
                "--beta-text",
                str(args.beta_text),
                "--beta-structure",
                str(args.beta_structure),
                "--lambda-parent",
                str(args.lambda_parent),
                "--lambda-depth",
                str(args.lambda_depth),
                "--lambda-next",
                str(args.lambda_next),
                "--lambda-child",
                str(args.lambda_child),
                "--lambda-language-alignment",
                str(alignment_lambda),
                "--lr",
                str(args.lr),
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--out",
                str(checkpoint),
            ]
            eval_command = [
                "python3",
                "-m",
                "tractatus_structure_latents.evaluation.evaluate_structure",
                "--data",
                str(args.data),
                "--checkpoint",
                str(checkpoint),
                "--batch-size",
                "64",
                "--latent-part",
                "structure",
                "--export-latents",
                str(latent_path),
            ]
            print(f"running {tag}/{seed_name}", flush=True)
            run_command(train_command, log_path=log_path, dry_run=args.dry_run)
            run_command(eval_command, stdout_path=metric_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
