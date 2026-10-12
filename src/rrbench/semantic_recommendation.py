from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from rrbench.io import sha256, write_json

DAY_MS = 86400000
WINDOWS = (
    (1, 1649347200000, 1650556800000),
    (2, 1650556800000, 1651161600000),
    (3, 1651161600000, 1652025600000),
)


@lru_cache(maxsize=64)
def declared_midnight(date: str) -> int:
    value = datetime.strptime(date, "%Y%m%d")
    if value.strftime("%Y%m%d") != date:
        raise ValueError("Declared interaction date must be an exact YYYYMMDD value.")
    return int(value.replace(tzinfo=timezone(timedelta(hours=8))).timestamp() * 1000)


def recorded_clock_envelope(date: str, timestamp: int) -> tuple[int, int, int]:
    midnight = declared_midnight(date)
    if not isinstance(timestamp, int) or timestamp <= 0:
        raise ValueError("Interaction timestamp must be positive integer epoch milliseconds.")
    phase = next((phase for phase, lower, upper in WINDOWS if lower <= midnight < upper), 0)
    if not phase:
        raise ValueError("Declared interaction date is outside the benchmark periods.")
    return min(midnight, timestamp), max(midnight + DAY_MS, timestamp + 1), phase


def admitted_phase(lower: int, upper: int, nominal_mask: int) -> int:
    return next((
        phase for phase, start, end in WINDOWS
        if nominal_mask == 1 << (phase - 1) and start <= lower and upper <= end
    ), 0)


def admit_timestamp_block(records: list[tuple[str, int]]) -> int:
    if not records:
        raise ValueError("A timestamp block must contain recorded interaction clocks.")
    envelopes = [recorded_clock_envelope(date, timestamp) for date, timestamp in records]
    mask = 0
    for _, _, phase in envelopes:
        mask |= 1 << (phase - 1)
    return admitted_phase(min(row[0] for row in envelopes), max(row[1] for row in envelopes), mask)


def audit_kuairand(
    directory: Path, output: Path, minimum_catalog: int = 1000000, rows_per_shard: int = 262144,
) -> dict[str, object]:
    if minimum_catalog <= 0 or rows_per_shard <= 0:
        raise ValueError("Catalog and shard row limits must be positive.")
    manifest_path = output / "corpus.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["minimum_catalog_gate"] != minimum_catalog or manifest["clock_policy_version"] != 2:
            raise ValueError("Recommendation catalog is bound to a different eligibility contract.")
        paths = [
            ("catalog.tsv", "catalog_sha256"), ("original_ids.npy", "ids_sha256"),
            ("catalog_offsets.npy", "offsets_sha256"),
        ]
        for filename, key in paths:
            if sha256(output / filename) != manifest[key]:
                raise ValueError("Audited recommendation catalog bytes changed.")
        for shard in manifest["shards"]:
            if sha256(output / shard["file"]) != shard["sha256"]:
                raise ValueError("Audited recommendation text shard bytes changed.")
        return manifest
    started = time.perf_counter()
    root = directory / "raw" / "kuairand" / "KuaiRand-1K"
    source = root / "data"
    captions = directory / "downloads" / "kuairand_video_captions.csv"
    output.mkdir(parents=True, exist_ok=True)
    database = output / "private-catalog.sqlite"
    if database.exists():
        raise ValueError("Incomplete audit already has a database; retain it and use a new version.")
    train = source / "log_standard_4_08_to_4_21_1k.csv"
    future = source / "log_standard_4_22_to_5_08_1k.csv"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            "CREATE TABLE blocks(user INTEGER,timestamp INTEGER,lower INTEGER,upper INTEGER,"
            "nominal_mask INTEGER,rows INTEGER,phase INTEGER DEFAULT 0,PRIMARY KEY(user,timestamp))"
        )
        connection.execute(
            "CREATE TABLE events(user INTEGER,item INTEGER,date INTEGER,timestamp INTEGER,label INTEGER)"
        )
        diagnostics, total_rows = {}, 0
        for filename in (train, future):
            mismatch, signed_day_delta, crossing, hours = Counter(), Counter(), Counter(), Counter()
            relative_sample, minimum_offset, maximum_offset = [], None, None
            events, blocks, count = [], [], 0

            def flush_rows() -> None:
                connection.executemany("INSERT INTO events VALUES (?,?,?,?,?)", events)
                connection.executemany(
                    "INSERT INTO blocks(user,timestamp,lower,upper,nominal_mask,rows) VALUES (?,?,?,?,?,1) "
                    "ON CONFLICT(user,timestamp) DO UPDATE SET lower=MIN(lower,excluded.lower),"
                    "upper=MAX(upper,excluded.upper),nominal_mask=nominal_mask|excluded.nominal_mask,rows=rows+1",
                    blocks,
                )
                connection.commit()
                events.clear()
                blocks.clear()

            with filename.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    timestamp = int(row["time_ms"])
                    lower, upper, phase = recorded_clock_envelope(row["date"], timestamp)
                    midnight = declared_midnight(row["date"])
                    user, item, label = int(row["user_id"]), int(row["video_id"]), int(row["long_view"])
                    if label not in (0, 1) or user < 0 or item < 0:
                        raise ValueError("Interaction has an invalid user/item/long-view value.")
                    events.append((user, item, int(row["date"]), timestamp, label))
                    blocks.append((user, timestamp, lower, upper, 1 << (phase - 1)))
                    offset = timestamp - midnight
                    minimum_offset = offset if minimum_offset is None else min(minimum_offset, offset)
                    maximum_offset = offset if maximum_offset is None else max(maximum_offset, offset)
                    signed_day_delta[str(offset // DAY_MS)] += 1
                    mismatch["utc8_date_matches"] += int(0 <= offset < DAY_MS)
                    utc_midnight = midnight + 8 * 3600000
                    mismatch["utc_date_matches"] += int(0 <= timestamp - utc_midnight < DAY_MS)
                    epoch_phase = next((p for p, a, b in WINDOWS if a <= timestamp < b), 0)
                    crossing[f"{phase}->{epoch_phase}"] += 1
                    hours[f"{row['hourmin']}->{(timestamp + 8 * 3600000) // 3600000 % 24}"] += 1
                    if count % 128 == 0:
                        relative_sample.append(offset)
                    count += 1
                    if len(events) == 100000:
                        flush_rows()
            if events:
                flush_rows()
            diagnostics[filename.name] = {
                "rows": count, "date_matches": dict(mismatch), "signed_utc8_day_difference": dict(signed_day_delta),
                "nominal_to_epoch_phase": dict(crossing), "declared_hourmin_to_epoch_hour": dict(hours),
                "timestamp_minus_declared_midnight_extrema_ms": [minimum_offset, maximum_offset],
                "timestamp_minus_declared_midnight_sample_quantiles_ms": np.quantile(
                    relative_sample, [0, 0.01, 0.5, 0.99, 1],
                ).tolist(),
                "quantile_sampling": "Every128th source row; clock-only diagnostic, not an eligibility estimate.",
            }
            total_rows += count
        for phase, lower, upper in WINDOWS:
            connection.execute(
                "UPDATE blocks SET phase=? WHERE nominal_mask=? AND lower>=? AND upper<=?",
                (phase, 1 << (phase - 1), lower, upper),
            )
        connection.execute("CREATE INDEX events_clock ON events(user,timestamp)")
        connection.execute(
            "CREATE TABLE observed AS SELECT e.item,MIN(e.timestamp) first_time FROM events e "
            "JOIN blocks b USING(user,timestamp) WHERE b.phase=1 GROUP BY e.item"
        )
        connection.execute("CREATE UNIQUE INDEX observed_item ON observed(item)")
        observed_ids = {row[0] for row in connection.execute("SELECT item FROM observed")}
        metadata_files = list(source.glob("video_features_basic*.csv"))
        if len(metadata_files) != 1:
            raise ValueError("Expected one official basic-video metadata file for ID-map verification.")
        metadata_ids = set()
        with metadata_files[0].open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                item = int(row["video_id"])
                if item in metadata_ids:
                    raise ValueError("Basic-video metadata contains duplicate item IDs.")
                metadata_ids.add(item)
        if not observed_ids.issubset(metadata_ids):
            raise ValueError("Training-observed video IDs do not all join official basic metadata.")
        metadata_count = len(metadata_ids)
        del metadata_ids
        connection.execute("CREATE TABLE semantic(item INTEGER PRIMARY KEY,text TEXT NOT NULL)")
        caption_rows, nonempty, duplicates, chunk = 0, 0, 0, []

        def flush_captions() -> None:
            nonlocal duplicates
            before = connection.total_changes
            connection.executemany("INSERT OR IGNORE INTO semantic VALUES (?,?)", chunk)
            duplicates += len(chunk) - (connection.total_changes - before)
            connection.commit()
            chunk.clear()

        with captions.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                caption_rows += 1
                item = int(row["final_video_id"])
                if item not in observed_ids:
                    continue
                text = " ".join(
                    value.strip().replace("\t", " ").replace("\n", " ").replace("\r", " ")
                    for value in (row["caption"], row["show_cover_text"]) if value.strip()
                ).strip()
                if not text:
                    continue
                nonempty += 1
                chunk.append((item, text))
                if len(chunk) == 50000:
                    flush_captions()
        if chunk:
            flush_captions()
        semantic_count = connection.execute("SELECT COUNT(*) FROM semantic").fetchone()[0]
        if semantic_count < minimum_catalog:
            raise ValueError("Post-guard training-observed semantic inventory fails the million-item gate.")
        if duplicates:
            raise ValueError("Caption join duplicates training-observed IDs; no revision may be silently selected.")
        original_ids = np.lib.format.open_memmap(
            output / "original_ids.npy", mode="w+", dtype=np.int64, shape=(semantic_count,),
        )
        offsets = np.lib.format.open_memmap(
            output / "catalog_offsets.npy", mode="w+", dtype=np.uint64, shape=(semantic_count + 1,),
        )
        digest, position, count, shard_stream, shards = hashlib.sha256(), 0, 0, None, []
        shard_digest = None
        with (output / "catalog.tsv").open("wb") as stream:
            for item, text in connection.execute("SELECT item,text FROM semantic ORDER BY item"):
                if count % rows_per_shard == 0:
                    if shard_stream is not None:
                        shard_stream.close()
                        shards[-1]["sha256"] = shard_digest.hexdigest()
                    filename = f"corpus_{len(shards):03d}.tsv"
                    shard_stream = (output / filename).open("wb")
                    shard_digest = hashlib.sha256()
                    shards.append({"file": filename, "start": count, "rows": 0})
                line = f"{count}\t{text}\n".encode("utf-8")
                original_ids[count], offsets[count] = item, position
                position += len(line)
                stream.write(line)
                digest.update(line)
                shard_stream.write(line)
                shard_digest.update(line)
                shards[-1]["rows"] += 1
                count += 1
        shard_stream.close()
        shards[-1]["sha256"] = shard_digest.hexdigest()
        offsets[count] = position
        offsets.flush()
        original_ids.flush()
        coverage = {}
        for phase, label in ((2, "validation"), (3, "test")):
            events, positives = connection.execute(
                "SELECT COUNT(*),COALESCE(SUM(e.label),0) FROM events e JOIN blocks b USING(user,timestamp) "
                "WHERE b.phase=?", (phase,),
            ).fetchone()
            targets = connection.execute(
                "SELECT COUNT(*) FROM events e JOIN blocks b USING(user,timestamp) JOIN semantic s USING(item) "
                "WHERE b.phase=? AND e.label=1", (phase,),
            ).fetchone()[0]
            coverage.update({
                label + "_events": events, label + "_positive_targets": positives,
                label + "_positive_targets_in_catalog": targets,
            })
        training_events, training_positives, training_users = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(e.label),0),COUNT(DISTINCT e.user) FROM events e "
            "JOIN blocks b USING(user,timestamp) WHERE b.phase=1"
        ).fetchone()
        clock_audit = {
            "files": diagnostics, "source_events": total_rows,
            "timestamp_blocks": connection.execute("SELECT COUNT(*) FROM blocks").fetchone()[0],
            "quarantined_blocks": connection.execute("SELECT COUNT(*) FROM blocks WHERE phase=0").fetchone()[0],
            "quarantined_events": connection.execute(
                "SELECT COALESCE(SUM(rows),0) FROM blocks WHERE phase=0"
            ).fetchone()[0],
            "maximum_timestamp_block_rows": connection.execute("SELECT MAX(rows) FROM blocks").fetchone()[0],
            "multi_day_timestamp_blocks": connection.execute(
                "SELECT COUNT(*) FROM blocks WHERE upper-lower>?", (DAY_MS,),
            ).fetchone()[0],
            "cross_nominal_period_timestamp_blocks": connection.execute(
                "SELECT COUNT(*) FROM blocks WHERE nominal_mask NOT IN (1,2,4)"
            ).fetchone()[0],
            "malformed_clock_policy": "Fail explicitly before catalog publication; no guessed timestamp/date repairs.",
        }
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    result = {
        "dataset": "KuaiRand-1K with official2026caption supplement", "rows": count, "shards": shards,
        "clock_policy_version": 2, "clock_audit": clock_audit,
        "training_events": training_events, "training_long_views": training_positives,
        "training_users": training_users, "metadata_unique_items": metadata_count,
        "training_observed_unique_items": len(observed_ids), "caption_source_rows": caption_rows,
        "nonempty_training_observed_caption_rows": nonempty, "caption_join_duplicates": duplicates,
        "usable_unique_semantic_catalog": semantic_count, "minimum_catalog_gate": minimum_catalog,
        "million_item_gate_pass": True, "catalog_sha256": digest.hexdigest(),
        "ids_sha256": sha256(output / "original_ids.npy"), "offsets_sha256": sha256(output / "catalog_offsets.npy"),
        "training_interactions_sha256": sha256(train), "heldout_interactions_sha256": sha256(future),
        "caption_source_sha256": sha256(captions), "original_license_sha256": sha256(root / "LICENSE"),
        "metadata_source_sha256": sha256(metadata_files[0]),
        "clock_contract": "Declared-date splits plus dual-recorded-clock envelope quarantine of whole(user,time_ms)blocks.",
        "timezone": "UTC+08benchmark convention, not a documentation-established exposure timezone.",
        "history_contract": "Frozen training-only exposure histories; history block upper<=target block lower. No feedback conditioning.",
        "train_dates": ["2022-04-08", "2022-04-21"], "validation_dates": ["2022-04-22", "2022-04-28"],
        "test_dates": ["2022-04-29", "2022-05-08"],
        "metadata_contract": "Fixed retrospective2026semantic captions; exact historical exposure/feedback availability unestablished.",
        "excluded_features": ["monthly aggregates", "future popularity", "visible_status", "future-aware k-core"],
        "license": "ArchiveCC-BY-SA-4.0 retained; supplementCC-BY-4.0; publish aggregate outputs/code only.",
        "coverage_counts": coverage, "audit_seconds": time.perf_counter() - started,
    }
    write_json(manifest_path, result)
    print(f"MILLION_ITEM_RECOMMENDATION_CATALOG_VERIFIED rows={semantic_count}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a real recommendation catalog with dual-clock boundary quarantine.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit_kuairand(Path(args.data), Path(args.output))


if __name__ == "__main__":
    main()
