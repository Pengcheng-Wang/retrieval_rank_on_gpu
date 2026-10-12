from __future__ import annotations

import argparse


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rrbench")
    commands = parser.add_subparsers(dest="command", required=True)
    ann = commands.add_parser("ann", help="ANN latency-recall sweeps on shared IVF partitions")
    common(ann)
    ann.add_argument("--items", type=int, default=1000000)
    ann.add_argument("--dimension", type=int, default=128)
    ann.add_argument("--queries", type=int, default=128)
    ann.add_argument("--distribution", choices=["isotropic", "clustered"], default="clustered")
    ann.add_argument("--clusters", type=int, default=1024)
    ann.add_argument("--ks", nargs="+", type=int, default=[100, 1024, 2048, 4096])
    ann.add_argument("--probes", nargs="+", type=int, default=[8, 24, 64, 256])
    ann.add_argument("--batches", nargs="+", type=int, default=[1, 16])
    ann.add_argument("--repeats", type=int, default=5)
    ann.add_argument("--backends", nargs="+", default=[
        "faiss-ivf-cpu", "faiss-ivf-gpu", "silver-fp32", "silver-fp16",
        "silver-int8-global", "silver-int8-rowwise",
    ])
    filtering = commands.add_parser("filter", help="Exact filtering and conservative Bloom audit")
    common(filtering)
    filtering.add_argument("--items", type=int, default=1000000)
    filtering.add_argument("--features", type=int, default=6)
    filtering.add_argument("--values", type=int, default=2)
    filtering.add_argument("--query-types", type=int, default=5)
    filtering.add_argument("--multipliers", nargs="+", type=float, default=[2.0, 5.0, 10.0])
    rank = commands.add_parser("rank-timing", help="Actual TorchRec/MLP runtime, not quality")
    common(rank)
    rank.add_argument("--users", type=int, default=100000)
    rank.add_argument("--items", type=int, default=1000000)
    rank.add_argument("--candidates", nargs="+", type=int, default=[100, 1024, 10000])
    rank.add_argument("--batches", nargs="+", type=int, default=[1, 16])
    rank.add_argument("--models", nargs="+", default=["mf", "mlp", "torchrec-dlrm", "torchrec-dcn"])
    quality = commands.add_parser("quality", help="Train and evaluate MovieLens recommendation models")
    common(quality)
    quality.add_argument("--data", required=True)
    quality.add_argument("--model", choices=["mf", "mlp", "torchrec-dlrm", "torchrec-dcn"], required=True)
    quality.add_argument("--epochs", type=int, default=12)
    quality.add_argument("--batch-size", type=int, default=4096)
    quality.add_argument("--learning-rate", type=float, default=0.003)
    quality.add_argument("--negatives", type=int, default=4)
    quality.add_argument("--eval-users", type=int, default=1000)
    quality.add_argument("--candidates", nargs="+", type=int, default=[100, 500, 1000])
    quality.add_argument("--mf-checkpoint")
    pipeline = commands.add_parser("pipeline", help="Local ANN/filter/rerank including interstage transfers")
    common(pipeline)
    pipeline.add_argument("--items", type=int, default=1000000)
    pipeline.add_argument("--queries", type=int, default=128)
    pipeline.add_argument("--clusters", type=int, default=1024)
    pipeline.add_argument("--candidates", nargs="+", type=int, default=[128, 512])
    pipeline.add_argument("--oversample", type=int, default=4)
    pipeline.add_argument("--probes", nargs="+", type=int, default=[8, 24, 64])
    pipeline.add_argument("--batches", nargs="+", type=int, default=[1, 16])
    pipeline.add_argument("--models", nargs="+", default=["mlp", "torchrec-dlrm", "torchrec-dcn"])
    pipeline.add_argument("--backends", nargs="+", default=[
        "faiss-ivf-cpu", "faiss-ivf-gpu", "silver-fp32",
        "silver-int8-global", "silver-int8-global-full",
    ])
    audit = commands.add_parser("audit-upstream", help="Measure known Bloom and INT8 correctness hazards")
    audit.add_argument("--output", required=True)
    stages = commands.add_parser("ann-stages", help="Isolated fused scoring versus gathered tensor scoring")
    common(stages)
    stages.add_argument("--items", type=int, default=1000000)
    stages.add_argument("--dimension", type=int, default=128)
    stages.add_argument("--queries", type=int, default=128)
    stages.add_argument("--distribution", choices=["isotropic", "clustered"], default="clustered")
    stages.add_argument("--clusters", type=int, default=1024)
    stages.add_argument("--probes", nargs="+", type=int, default=[24, 64, 256])
    stages.add_argument("--batches", nargs="+", type=int, default=[16])
    stages.add_argument("--repeats", type=int, default=3)
    trained = commands.add_parser("quality-ann", help="Frozen learned models with actual ANN retrieval")
    common(trained)
    trained.add_argument("--data", required=True)
    trained.add_argument("--checkpoints", required=True)
    trained.add_argument("--eval-users", type=int, default=2000)
    trained.add_argument("--clusters", type=int, default=64)
    trained.add_argument("--probes", nargs="+", type=int, default=[8, 24, 64])
    trained.add_argument("--candidates", nargs="+", type=int, default=[100, 500, 1000])
    trained.add_argument("--backends", nargs="+", default=[
        "faiss-ivf-cpu", "faiss-ivf-gpu", "silver-fp32", "silver-fp16", "silver-int8-global", "silver-int8-rowwise",
    ])
    profile = commands.add_parser("pipeline-profile", help="Measure exact-verification bottleneck and equivalent batching")
    common(profile)
    profile.add_argument("--input", required=True)
    args = parser.parse_args()
    if args.command == "ann":
        from rrbench.benchmarks import run_ann
        run_ann(args)
    elif args.command == "filter":
        from rrbench.benchmarks import run_filter
        run_filter(args)
    elif args.command == "rank-timing":
        from rrbench.benchmarks import run_rank_timing
        run_rank_timing(args)
    elif args.command == "pipeline":
        from rrbench.pipeline import run_pipeline
        run_pipeline(args)
    elif args.command == "audit-upstream":
        from pathlib import Path
        from rrbench.audit import audit_upstream
        print(audit_upstream(Path(args.output)))
    elif args.command == "ann-stages":
        from rrbench.stages import run_stages
        run_stages(args)
    elif args.command == "quality-ann":
        from rrbench.quality_ann import run_quality_ann
        run_quality_ann(args)
    elif args.command == "pipeline-profile":
        from rrbench.pipeline_profile import profile_pipeline
        profile_pipeline(args)
    else:
        from rrbench.quality import run_quality
        run_quality(args)


if __name__ == "__main__":
    main()
