from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import re
import subprocess
import time

from rrbench.io import write_json
from rrbench.process_lease import guard_worker_parent, read_process, wait_for_start_gate
from rrbench.runtime import require_cuda


def validate_scope(
    physical_device: int, expected_devices: int, visible: str, device_uuid: str,
    benchmark_device: int = 0,
) -> None:
    if (
        any(type(value) is not int for value in (physical_device, expected_devices, benchmark_device))
        or expected_devices <= 1 or not 0 <= physical_device < expected_devices
        or not 0 <= benchmark_device < expected_devices or physical_device == benchmark_device
    ):
        raise ValueError("Administrative context must use one explicitly unused device.")
    if re.fullmatch(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", device_uuid) is None:
        raise ValueError("Administrative context requires the full intended device UUID.")
    if visible != device_uuid:
        raise ValueError("Reservation context must have only its administrative GPU visible.")


def cooperative_ready(gpu_clients: list[int], owner_pid: int) -> bool:
    if type(owner_pid) is not int or owner_pid <= 0 or any(type(pid) is not int or pid <= 0 for pid in gpu_clients):
        raise ValueError("Cooperative admission needs valid observed and owned PIDs.")
    return bool(gpu_clients) and set(gpu_clients) == {owner_pid}


def deadline_timestamp(value: str) -> float:
    deadline = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("Reservation deadline must include a timezone.")
    return deadline.timestamp()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded administrative GPU context for an existing cooperative supervisor.")
    parser.add_argument("--physical-device", type=int, default=7)
    parser.add_argument("--expected-devices", type=int, default=8)
    parser.add_argument("--benchmark-device", type=int, default=0)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--parent-start", type=int, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    validate_scope(
        args.physical_device, args.expected_devices, os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        args.device_uuid, args.benchmark_device,
    )
    end = deadline_timestamp(args.deadline)
    if time.time() >= end:
        raise RuntimeError("Administrative context deadline has already expired.")
    guard_worker_parent(args.parent, args.parent_start)
    wait_for_start_gate(args.gate, args.owner, args.physical_device, args.parent, args.parent_start)
    mapping = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=15,
    )
    devices = {int(index.strip()): uuid.strip() for index, uuid in (row.split(",") for row in mapping.splitlines())}
    if len(devices) != args.expected_devices or devices.get(args.physical_device) != args.device_uuid:
        raise RuntimeError("Administrative device map changed before CUDA initialization.")
    import torch

    device = require_cuda()
    context = torch.empty(1, dtype=torch.uint8, device=device)
    torch.cuda.synchronize(device)
    write_json(args.ready, {
        "complete": True, "pid": os.getpid(), "physical_device": args.physical_device,
        "benchmark_device": args.benchmark_device, "context_tensor_bytes": context.numel(),
        "device_uuid": args.device_uuid,
        "deadline_utc": args.deadline,
        "scope": "Administrative idle context only; no background compute, benchmark occupancy, throughput or serving claim.",
    })
    while time.time() < end:
        parent, state = read_process(args.parent, Path("/proc"))
        if parent["start_ticks"] != args.parent_start or state in ("Z", "X"):
            raise RuntimeError("Administrative context parent exited or changed.")
        time.sleep(min(5, max(0.01, end - time.time())))
    print("ADMINISTRATIVE_CONTEXT_ABSOLUTE_DEADLINE_EXIT", flush=True)


if __name__ == "__main__":
    main()
