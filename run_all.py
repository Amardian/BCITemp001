#!/usr/bin/env python3
"""Run the complete NAC pipeline (paper reimplementation).

Examples
--------
    python run_all.py --preset smoke                      # 2-4 min check
    python run_all.py --preset demo                       # full demo run
    python run_all.py --preset demo --datasets karaone
    python run_all.py --preset demo --stages teacher,nap  # checkpointed
    python run_all.py --preset paper --data-dir /path/to/datasets

Stage checkpoints (`results/ckpt_<ds>.pth`) let long runs be split
across multiple invocations; `--stages all` (default) runs everything.

Outputs: results/tables/*.csv, results/figures/*.png, results/*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="demo",
                    choices=["smoke", "demo", "paper"])
    ap.add_argument("--datasets", default="karaone,bci2a")
    ap.add_argument("--stages", default="all",
                    help="teacher,nap,distill,baselines,tables,xai,expert,"
                         "bench,stats or 'all'")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="",
                    help="optional directory with real datasets")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    from nac.config import get_preset
    from nac.pipeline import STAGE_ORDER, make_logger, run_dataset

    cfg = get_preset(args.preset, seed=args.seed, outdir=args.outdir,
                     data_dir=args.data_dir, torch_threads=args.threads)
    torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)

    os.makedirs(args.outdir, exist_ok=True)
    logger = make_logger(os.path.join(args.outdir, "run.log"))
    stages = (STAGE_ORDER if args.stages == "all"
              else [s.strip() for s in args.stages.split(",") if s.strip()])
    logger(f"NAC pipeline run  preset={cfg.preset}  seed={cfg.seed}  "
           f"datasets={args.datasets}  stages={stages}  "
           f"torch={torch.__version__}  threads={cfg.torch_threads}  "
           f"start={time.strftime('%Y-%m-%d %H:%M:%S')}")

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        run_dataset(ds, cfg, args.outdir, logger, stages=stages)

    # merge per-dataset summaries when they exist
    merged = {}
    for p in sorted(glob.glob(os.path.join(args.outdir, "summary_*.json"))):
        if p.endswith("summary.json"):
            continue
        with open(p) as f:
            merged[os.path.basename(p).replace("summary_", "")
                   .replace(".json", "")] = json.load(f)
    if merged:
        with open(os.path.join(args.outdir, "summary.json"), "w") as f:
            json.dump(merged, f, indent=2, default=str)
        logger("\n================ FINAL SUMMARY ================")
        for ds, s in merged.items():
            nac, teacher, exp = s["nac"], s["teacher"], s["expert_layer"]
            logger(
                f"[{ds}] teacher {teacher['accuracy']:.1f}% @ "
                f"{teacher['size_mb']:.2f}MB  ->  NAC {nac['accuracy']:.1f}% "
                f"@ {nac['size_mb']:.2f}MB (-{nac['size_reduction_pct']:.1f}%) | "
                f"CSI={nac['csi']:.3f} | SPR={nac['spr']:.3f} | "
                f"coverage={exp['conformal_coverage']:.1f}% @ "
                f"committed-acc={exp['committed_accuracy']:.1f}% | "
                f"latency={nac['latency_ms']:.1f}ms | "
                f"online-acc={100 * nac['online_accuracy']:.1f}%")
        logger("All outputs -> " + os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
