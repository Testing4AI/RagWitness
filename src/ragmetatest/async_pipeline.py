"""Durable local ingestion queue and independently runnable recovery worker.

This is an experimental local pipeline, not an upstream or production service.
The worker always reconciles to the latest committed source revision. Queue
redelivery is safe because writes replace a document using stable chunk IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from ragmetatest.retrievers import ContrieverEncoder, QdrantContrieverRetriever
from ragmetatest.types import Document


class DurableQueue:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS source (
                doc_id TEXT PRIMARY KEY, version INTEGER, payload TEXT, deleted INTEGER
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, doc_id TEXT,
                version INTEGER, status TEXT DEFAULT 'pending', lease_until REAL DEFAULT 0,
                attempts INTEGER DEFAULT 0, error TEXT, completed_at REAL
            );
            CREATE TABLE IF NOT EXISTS applied (doc_id TEXT PRIMARY KEY, version INTEGER);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, stamp REAL, pid INTEGER,
                kind TEXT, job_id INTEGER, detail TEXT
            );
        """)

    def log(self, kind, job_id=None, detail=""):
        self.connection.execute(
            "INSERT INTO events(stamp,pid,kind,job_id,detail) VALUES(?,?,?,?,?)",
            (time.time(), os.getpid(), kind, job_id, detail),
        )
        self.connection.commit()

    def publish(self, event_id: str, document: Document, *, deleted=False):
        with self.connection:
            prior = self.connection.execute(
                "SELECT version FROM source WHERE doc_id=?", (document.doc_id,),
            ).fetchone()
            if prior is None or document.version >= prior["version"]:
                self.connection.execute(
                    "INSERT OR REPLACE INTO source VALUES(?,?,?,?)",
                    (document.doc_id, document.version, json.dumps(asdict(document)), int(deleted)),
                )
            cursor = self.connection.execute(
                "INSERT INTO jobs(event_id,doc_id,version) VALUES(?,?,?)",
                (event_id, document.doc_id, document.version),
            )
        return cursor.lastrowid

    def claim(self, lease_seconds=3):
        self.connection.execute("BEGIN IMMEDIATE")
        row = self.connection.execute(
            "SELECT * FROM jobs WHERE status='pending' OR "
            "(status='processing' AND lease_until<?) ORDER BY id LIMIT 1", (time.time(),),
        ).fetchone()
        if row:
            self.connection.execute(
                "UPDATE jobs SET status='processing',lease_until=?,attempts=attempts+1 WHERE id=?",
                (time.time() + lease_seconds, row["id"]),
            )
        self.connection.commit()
        return dict(row) if row else None

    def snapshot(self):
        return [Document(**json.loads(row["payload"])) for row in self.connection.execute(
            "SELECT payload FROM source WHERE deleted=0 ORDER BY doc_id"
        )]

    def counts(self):
        return dict(self.connection.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))

    def settled(self):
        counts = self.counts()
        return not any(counts.get(key, 0) for key in ("pending", "processing", "failed"))

    def wait_settled(self, timeout=90):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.settled():
                return True
            time.sleep(0.05)
        return False


def worker(db_path, collection, url, *, request_timeout=10, ack_delay=0.0):
    from qdrant_client import QdrantClient

    queue = DurableQueue(db_path)
    encoder = ContrieverEncoder(batch_size=16)
    retriever = QdrantContrieverRetriever(
        collection_name=collection, encoder=encoder,
        client=QdrantClient(url=url, timeout=request_timeout, check_compatibility=False),
    )
    queue.log("worker_ready")
    while True:
        job = queue.claim()
        if job is None:
            time.sleep(0.05)
            continue
        try:
            queue.log("claimed", job["id"])
            source = queue.connection.execute(
                "SELECT * FROM source WHERE doc_id=?", (job["doc_id"],),
            ).fetchone()
            applied = queue.connection.execute(
                "SELECT version FROM applied WHERE doc_id=?", (job["doc_id"],),
            ).fetchone()
            if applied is None or applied["version"] < source["version"]:
                if source["deleted"]:
                    retriever.delete_document(source["doc_id"])
                else:
                    retriever.replace_document(Document(**json.loads(source["payload"])))
                queue.log("vector_written", job["id"], str(source["version"]))
                # An explicit timing knob permits an external process kill to
                # target the acknowledgement boundary; no faulty branch is used.
                if ack_delay:
                    time.sleep(ack_delay)
            with queue.connection:
                queue.connection.execute(
                    "INSERT OR REPLACE INTO applied VALUES(?,?)",
                    (source["doc_id"], source["version"]),
                )
                queue.connection.execute(
                    "UPDATE jobs SET status='done',completed_at=?,error=NULL WHERE id=?",
                    (time.time(), job["id"]),
                )
            queue.log("acknowledged", job["id"])
        except Exception as error:  # noqa: BLE001 - journal every worker failure, never count it as a pass.
            with queue.connection:
                queue.connection.execute(
                    "UPDATE jobs SET status=CASE WHEN attempts>=5 THEN 'failed' ELSE 'pending' END,"
                    "error=? WHERE id=?", (type(error).__name__, job["id"]),
                )
            queue.log("retry", job["id"], type(error).__name__)
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--ack-delay", type=float, default=0)
    args = parser.parse_args()
    worker(args.db, args.collection, args.url, request_timeout=args.timeout, ack_delay=args.ack_delay)


if __name__ == "__main__":
    main()
