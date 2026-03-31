#!/usr/bin/env python3
"""Autoresearch experiment entry point."""
import json
import sys
from pathlib import Path

project_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_dir))

from ml.experiment import run_experiment

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gp")
    parser.add_argument("--data", default=str(project_dir / "results" / "sweep_all.json"))
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", default="64,64,32")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--output", default="autoresearch/latest_result.json")
    args = parser.parse_args()

    kwargs = {}
    if args.model == "mlp":
        kwargs = {"lr": args.lr, "epochs": args.epochs,
                  "hidden_layers": [int(x) for x in args.hidden.split(",")]}
    elif args.model == "linear":
        kwargs = {"degree": args.degree}
    elif args.model == "pinn":
        kwargs = {"epochs": args.epochs, "physics_weight": args.physics_weight}

    result = run_experiment(args.model, args.data, model_kwargs=kwargs)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(f"Result saved: {args.output}")

    # Print score for Autoresearch to parse
    te = result.get("test_eval", {})
    score = (te.get("r2_cd", 0) + te.get("r2_cl", 0) + te.get("r2_ld_ratio", 0)) / 3
    print(f"SCORE: {score:.4f}")

if __name__ == "__main__":
    main()
