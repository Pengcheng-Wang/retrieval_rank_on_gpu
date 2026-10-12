import io
import tarfile

import numpy as np
import pytest

import rrbench.semantic_data as module


def test_actual_corpus_audit_requires_all_unique_ordered_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PASSAGE_COUNT", 4)
    source = tmp_path / "collection.tsv"
    source.write_bytes(b"0\tfirst passage\n1\tsecond\n2\tthird\n3\tfourth\n")
    result = module.shard_passages(source, tmp_path / "prepared", rows_per_shard=2)
    assert result["rows"] == result["unique_ids"] == 4
    assert [row["rows"] for row in result["shards"]] == [2, 2]
    assert len(np.load(tmp_path / "prepared" / "passage_offsets.npy")) == 5
    corrupted = tmp_path / "bad.tsv"
    corrupted.write_bytes(b"0\tfirst\n0\tduplicate\n2\tthird\n3\tfourth\n")
    with pytest.raises(ValueError, match="unique ordered"):
        module.shard_passages(corrupted, tmp_path / "bad")


def test_dataset_archive_cannot_escape_the_explicit_destination(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("../outside.txt")
        member.size = 3
        stream.addfile(member, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="unsafe"):
        module.safe_extract(archive, tmp_path / "safe")
    assert not (tmp_path / "outside.txt").exists()


def test_drive_qualified_archive_member_is_rejected_even_on_linux(tmp_path):
    archive = tmp_path / "drive.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("C:/outside.txt")
        member.size = 3
        stream.addfile(member, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="unsafe"):
        module.safe_extract(archive, tmp_path / "safe")


def test_prepared_corpus_cannot_reuse_missing_offsets(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PASSAGE_COUNT", 2)
    source = tmp_path / "collection.tsv"
    source.write_bytes(b"0\tfirst\n1\tsecond\n")
    destination = tmp_path / "prepared"
    module.shard_passages(source, destination)
    (destination / "passage_offsets.npy").unlink()
    with pytest.raises(ValueError, match="offsets"):
        module.shard_passages(source, destination)
