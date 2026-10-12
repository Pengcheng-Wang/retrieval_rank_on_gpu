from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np

from rrbench.io import bind_run_config, sha256, write_json
from rrbench.semantic_ir import judged_metrics


def validate_index_prefix(connection: sqlite3.Connection, expected_rows: int, integrity: bool = True) -> None:
    count, minimum, maximum = connection.execute("SELECT COUNT(*),MIN(rowid),MAX(rowid) FROM passages").fetchone()
    if (
        count != expected_rows or (expected_rows and (minimum != 1 or maximum != expected_rows))
        or (not expected_rows and (minimum is not None or maximum is not None))
    ):
        raise ValueError("BM25 resumed document IDs are not the exact committed1..progress prefix.")
    if integrity:
        connection.execute("INSERT INTO passages(passages) VALUES ('integrity-check')")


def unicode61_terms(queries: list[str]) -> list[list[str]]:
    terms = []
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE VIRTUAL TABLE query_text USING fts5(text,tokenize='unicode61 remove_diacritics 2')"
        )
        connection.execute("CREATE VIRTUAL TABLE query_tokens USING fts5vocab(query_text,'instance')")
        for query in queries:
            connection.execute("DELETE FROM query_text")
            connection.execute("INSERT INTO query_text(rowid,text) VALUES (1,?)", (query,))
            values = [
                row[0] for row in connection.execute(
                    "SELECT term FROM query_tokens WHERE doc=1 GROUP BY term ORDER BY MIN(offset)"
                )
            ]
            if not values:
                raise ValueError("BM25 query has no searchable Unicode61 tokens.")
            terms.append(values)
    return terms


def build_bm25(source: Path, output: Path, expected_rows: int = 8841823) -> dict[str, Any]:
    if expected_rows <= 0:
        raise ValueError("BM25 full-corpus count must be positive.")
    binding = {
        "source_sha256": sha256(source), "expected_rows": expected_rows,
        "engine": "SQLiteFTS5BM25", "tokenizer": "unicode61 remove_diacritics2",
        "k1": 1.2, "b": 0.75, "document_columns": ["body"], "no_stemming": True,
    }
    bind_run_config(output, binding)
    database = output / "index.sqlite"
    if (output / "completed.json").exists():
        completed = json.loads((output / "completed.json").read_text(encoding="utf-8"))
        if sha256(database) != completed["index_sha256"]:
            raise ValueError("Complete BM25 index bytes changed.")
        with sqlite3.connect(database) as connection:
            validate_index_prefix(connection, expected_rows)
        if sha256(database) != completed["index_sha256"]:
            raise ValueError("BM25 integrity verification unexpectedly changed the completed index bytes.")
        return completed
    started = time.perf_counter()
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA cache_size=-1048576")
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(body,content='',tokenize='unicode61 remove_diacritics 2')"
        )
        connection.execute("CREATE TABLE IF NOT EXISTS progress(id INTEGER PRIMARY KEY CHECK(id=1),rows INTEGER)")
        prior = connection.execute("SELECT rows FROM progress WHERE id=1").fetchone()
        completed_rows = prior[0] if prior else 0
        if not 0 <= completed_rows <= expected_rows:
            raise ValueError("BM25 progress is outside the declared corpus bounds.")
        validate_index_prefix(connection, completed_rows)
        count, values = 0, []
        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                identifier, separator, text = line.partition("\t")
                if not separator or int(identifier) != count or not text.strip():
                    raise ValueError("BM25 ingestion must preserve every ordered nonempty corpus ID.")
                if count >= completed_rows:
                    values.append((count + 1, text.rstrip("\n")))
                count += 1
                if len(values) == 100000:
                    connection.executemany("INSERT INTO passages(rowid,body) VALUES (?,?)", values)
                    connection.execute("INSERT INTO progress VALUES (1,?) ON CONFLICT(id) DO UPDATE SET rows=excluded.rows", (count,))
                    connection.commit()
                    values.clear()
                    print(f"BM25_FULL_CORPUS_INDEX rows={count}/{expected_rows}", flush=True)
        if count != expected_rows:
            raise ValueError("BM25 corpus row count differs from the full declared inventory.")
        if values:
            connection.executemany("INSERT INTO passages(rowid,body) VALUES (?,?)", values)
            connection.execute("INSERT INTO progress VALUES (1,?) ON CONFLICT(id) DO UPDATE SET rows=excluded.rows", (count,))
            connection.commit()
        validate_index_prefix(connection, expected_rows)
        connection.execute("INSERT INTO passages(passages) VALUES ('optimize')")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    completed = {
        "complete": True, "catalog_items": count, "index_sha256": sha256(database),
        "elapsed_seconds": time.perf_counter() - started, "binding": binding,
        "index_storage_bytes": database.stat().st_size,
        "boundary": "ReproducibleFTS5BM25 reference; not an exactPyserini/Anserini tokenizer replica.",
    }
    write_json(output / "completed.json", completed)
    return completed


def search_bm25(index: Path, queries: list[str], k: int) -> np.ndarray:
    if not queries or k <= 0:
        raise ValueError("BM25 needs nonempty queries and a positive candidate cap.")
    complete = json.loads((index / "completed.json").read_text(encoding="utf-8"))
    if not complete["complete"]:
        raise ValueError("BM25 index must complete before queries are evaluated.")
    candidates = np.full((len(queries), k), -1, dtype=np.int64)
    uri = (index / "index.sqlite").resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        validate_index_prefix(connection, complete["catalog_items"], integrity=False)
        for row, terms in enumerate(unicode61_terms(queries)):
            expression = " OR ".join('"' + term + '"' for term in terms)
            found = connection.execute(
                "SELECT rowid-1 FROM passages WHERE passages MATCH ? ORDER BY bm25(passages),rowid LIMIT ?",
                (expression, k),
            ).fetchall()
            candidates[row, :len(found)] = [item[0] for item in found]
    return candidates


def evaluate_bm25(index: Path, panel_file: Path, output: Path, k: int = 1000) -> dict[str, Any]:
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    completed_index = json.loads((index / "completed.json").read_text(encoding="utf-8"))
    if sha256(index / "index.sqlite") != completed_index["index_sha256"]:
        raise ValueError("BM25 index differs from the frozen complete receipt.")
    binding = {
        "index_receipt_sha256": sha256(index / "completed.json"), "panel_sha256": sha256(panel_file),
        "k": k, "candidate_policy": "Full-catalog lexicalretrievalwithoutpositivesinjected.",
        "query_tokenizer": "ActualFTS5Unicode61viafts5vocab; literalORterms.",
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        completed = json.loads((output / "completed.json").read_text(encoding="utf-8"))
        if sha256(output / "ids.npy") != completed["ids_sha256"]:
            raise ValueError("BM25 candidate predictions changed.")
        return completed
    started = time.perf_counter()
    ids = search_bm25(index, panel["queries"], k)
    elapsed = time.perf_counter() - started
    np.save(output / "ids.npy", ids)
    qrels = {query: {int(item): int(grade) for item, grade in values.items()} for query, values in panel["qrels"].items()}
    completed = {
        "complete": True, "queries": len(ids), "query_ids": panel["query_ids"], "ids_sha256": sha256(output / "ids.npy"),
        "relevance": judged_metrics(panel["query_ids"], ids, qrels, panel["binary_threshold"]),
        "elapsed_seconds": elapsed, "binding": binding, "positive_injection": False,
    }
    write_json(output / "completed.json", completed)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-catalog SQLiteFTS5BM25 lexical reference.")
    commands = parser.add_subparsers(dest="operation", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source", required=True)
    build.add_argument("--output", required=True)
    query = commands.add_parser("query")
    query.add_argument("--index", required=True)
    query.add_argument("--panel", required=True)
    query.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.operation == "build":
        build_bm25(Path(args.source), Path(args.output))
    else:
        evaluate_bm25(Path(args.index), Path(args.panel), Path(args.output))


if __name__ == "__main__":
    main()
