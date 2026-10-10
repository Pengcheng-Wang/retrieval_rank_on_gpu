import csv
from itertools import permutations

from rrbench.semantic_recommendation import admit_timestamp_block, audit_kuairand, recorded_clock_envelope


def test_within_training_midnight_clock_disagreement_is_not_a_future_event():
    records = [("20220419", 1650297250048)]
    assert admit_timestamp_block(records) == 1
    lower, upper, _ = recorded_clock_envelope(*records[0])
    assert lower < 1650297600000 < upper


def test_cutoff_crossing_is_quarantined_in_both_directions():
    assert admit_timestamp_block([("20220422", 1650556799000)]) == 0
    assert admit_timestamp_block([("20220421", 1650556801000)]) == 0


def test_tied_block_spanning_periods_is_never_split_by_row_order():
    records = [("20220421", 1650556799000), ("20220422", 1650556799000)]
    for ordering in permutations(records):
        assert admit_timestamp_block(list(ordering)) == 0


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_audited_catalog_excludes_boundary_block_and_is_label_independent(tmp_path):
    data = tmp_path / "input"
    root = data / "raw" / "kuairand" / "KuaiRand-1K"
    source = root / "data"
    source.mkdir(parents=True)
    (data / "downloads").mkdir()
    (root / "LICENSE").write_text("CC-BY-SA-4.0", encoding="utf-8")
    write_csv(source / "video_features_basic_1k.csv", ["video_id"], [
        {"video_id": item} for item in (11, 12, 13, 14)
    ])
    columns = ["user_id", "video_id", "date", "hourmin", "time_ms", "long_view"]
    training = [
        {"user_id": 0, "video_id": 11, "date": 20220419, "hourmin": 0, "time_ms": 1650297250048, "long_view": 0},
        {"user_id": 0, "video_id": 12, "date": 20220420, "hourmin": 1200, "time_ms": 1650427200000, "long_view": 1},
        {"user_id": 0, "video_id": 13, "date": 20220421, "hourmin": 2359, "time_ms": 1650556799000, "long_view": 1},
    ]
    future = [
        {"user_id": 0, "video_id": 14, "date": 20220422, "hourmin": 0, "time_ms": 1650556799000, "long_view": 1},
        {"user_id": 0, "video_id": 11, "date": 20220429, "hourmin": 1200, "time_ms": 1651204800000, "long_view": 1},
    ]
    train_path = source / "log_standard_4_08_to_4_21_1k.csv"
    write_csv(train_path, columns, training)
    write_csv(source / "log_standard_4_22_to_5_08_1k.csv", columns, future)
    write_csv(data / "downloads" / "kuairand_video_captions.csv",
              ["final_video_id", "caption", "show_cover_text"], [
                  {"final_video_id": item, "caption": f"video {item}", "show_cover_text": ""}
                  for item in (11, 12, 13, 14)
              ])
    first = audit_kuairand(data, tmp_path / "first", minimum_catalog=2)
    assert first["usable_unique_semantic_catalog"] == 2
    assert first["clock_audit"]["quarantined_events"] == 2
    assert first["coverage_counts"]["test_positive_targets_in_catalog"] == 1
    assert first["shards"][0]["rows"] == first["rows"] == 2
    for row in training:
        row["long_view"] = 1 - row["long_view"]
    write_csv(train_path, columns, reversed(training))
    second = audit_kuairand(data, tmp_path / "second", minimum_catalog=2)
    assert first["catalog_sha256"] == second["catalog_sha256"]
