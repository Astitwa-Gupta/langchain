# namespace_checkpointer.py
import json
import uuid
import asyncio
from datetime import datetime
from typing import Any, Optional, Sequence, Tuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.runnables import RunnableConfig

from models import CheckpointRow
from serde import to_serializable

# Per-thread (conversation) write locks to preserve integrity
_thread_locks: dict[str, asyncio.Lock] = {}

def _lock_for_thread(thread_id: str) -> asyncio.Lock:
    lock = _thread_locks.get(thread_id)
    if lock is None:
        lock = asyncio.Lock()
        _thread_locks[thread_id] = lock
    return lock

class AsyncSQLServerNamespaceCheckpointer(BaseCheckpointSaver):
    """
    Async SQL Server checkpointer:
    - isolates rows by (thread_id, checkpoint_ns) to avoid parallel write contention
    - serializes writes per thread_id for integrity
    - uses MERGE for upsert
    """

    def __init__(self, session_factory):
        super().__init__()
        self._session_factory = session_factory

    def _ns_from_config(self, config: dict, metadata: Optional[dict] = None) -> Optional[str]:
        cfg = (config or {}).get("configurable", {})
        return cfg.get("checkpoint_ns") or (metadata or {}).get("namespace")

    # ===== READ LATEST (for thread/ns)
    async def aget_tuple(self, config: RunnableConfig, *args, **kwargs):
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError("aget_tuple requires configurable.thread_id")
        checkpoint_ns = self._ns_from_config(config)

        async with self._session_factory() as sess:  # type: AsyncSession
            stmt = (
                select(CheckpointRow)
                .where(CheckpointRow.thread_id == thread_id)
                .where(CheckpointRow.checkpoint_ns == checkpoint_ns)
                .order_by(CheckpointRow.updated_at.desc())
                .limit(1)
            )
            res = await sess.execute(stmt)
            row = res.scalar_one_or_none()
            if not row:
                return None

            checkpoint = json.loads(row.checkpoint_blob) if row.checkpoint_blob else None
            metadata   = json.loads(row.metadata_blob) if row.metadata_blob else None
            writes     = json.loads(row.writes_json)   if row.writes_json else None

            # Return a minimal tuple-like dict compatible with most consumers
            return {"config": config, "checkpoint": checkpoint, "metadata": metadata, "writes": writes}

    # ===== WRITE FULL CHECKPOINT
    async def aput(self, config: RunnableConfig, checkpoint: Any, metadata: Any, new_versions: Any, *args, **kwargs) -> RunnableConfig:
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError("aput requires configurable.thread_id")
        checkpoint_ns = self._ns_from_config(config, metadata=metadata or {})
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id") or str(uuid.uuid4())

        cp_json   = json.dumps(to_serializable(checkpoint))
        meta_json = json.dumps(to_serializable(metadata or {}))
        now_iso   = datetime.utcnow().isoformat()

        async with self._session_factory() as sess:
            # Serialize writes per-thread to avoid interleaving
            async with _lock_for_thread(thread_id):
                # SQL Server MERGE upsert
                merge_sql = text("""
                    MERGE langgraph_checkpoints WITH (HOLDLOCK) AS tgt
                    USING (SELECT :thread_id AS thread_id, :checkpoint_ns AS checkpoint_ns) AS src
                    ON (tgt.thread_id = src.thread_id AND
                        ((tgt.checkpoint_ns IS NULL AND src.checkpoint_ns IS NULL) OR tgt.checkpoint_ns = src.checkpoint_ns))
                    WHEN MATCHED THEN
                      UPDATE SET checkpoint_id   = :checkpoint_id,
                                 checkpoint_blob = :checkpoint_blob,
                                 metadata_blob   = :metadata_blob,
                                 updated_at      = SYSUTCDATETIME()
                    WHEN NOT MATCHED THEN
                      INSERT (thread_id, checkpoint_ns, checkpoint_id, checkpoint_blob, metadata_blob, writes_json, updated_at)
                      VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :checkpoint_blob, :metadata_blob, N'[]', SYSUTCDATETIME());
                """)
                await sess.execute(
                    merge_sql,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_blob": cp_json,
                        "metadata_blob": meta_json,
                    },
                )
                await sess.commit()

        return config  # returning as-is is fine

    # ===== APPEND INCREMENTAL WRITES
    async def aput_writes(self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str, task_path: str = "", *args, **kwargs):
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError("aput_writes requires configurable.thread_id")
        checkpoint_ns = self._ns_from_config(config, metadata=kwargs.get("metadata") or {})

        # Serialize writes first (outside lock)
        to_append = []
        for idx, (channel, value) in enumerate(writes):
            to_append.append({
                "channel": channel,
                "task_id": task_id,
                "task_path": task_path,
                "idx": idx,
                "value": to_serializable(value),
                "ts": datetime.utcnow().isoformat()
            })
        writes_json = json.dumps(to_append)

        async with self._session_factory() as sess:
            async with _lock_for_thread(thread_id):
                # Ensure row exists, then append JSON array on server side
                # 1) create row if missing via MERGE
                merge_sql = text("""
                    MERGE langgraph_checkpoints WITH (HOLDLOCK) AS tgt
                    USING (SELECT :thread_id AS thread_id, :checkpoint_ns AS checkpoint_ns) AS src
                    ON (tgt.thread_id = src.thread_id AND
                        ((tgt.checkpoint_ns IS NULL AND src.checkpoint_ns IS NULL) OR tgt.checkpoint_ns = src.checkpoint_ns))
                    WHEN NOT MATCHED THEN
                      INSERT (thread_id, checkpoint_ns, checkpoint_id, checkpoint_blob, metadata_blob, writes_json, updated_at)
                      VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, NULL, NULL, N'[]', SYSUTCDATETIME());
                """)
                await sess.execute(
                    merge_sql,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": str(uuid.uuid4()),
                    },
                )
                # 2) append writes (JSON array concat)
                # SQL Server doesn't have native JSON array concat; read-append-write safely inside lock.
                select_sql = text("""
                    SELECT writes_json FROM langgraph_checkpoints
                    WHERE thread_id = :thread_id AND
                          ((checkpoint_ns IS NULL AND :checkpoint_ns IS NULL) OR checkpoint_ns = :checkpoint_ns)
                """)
                res = await sess.execute(select_sql, {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns})
                current = res.scalar_one()
                existing = json.loads(current or "[]")
                existing.extend(json.loads(writes_json))
                new_json = json.dumps(existing)

                update_sql = text("""
                    UPDATE langgraph_checkpoints
                       SET writes_json = :writes_json,
                           updated_at  = SYSUTCDATETIME()
                     WHERE thread_id = :thread_id AND
                           ((checkpoint_ns IS NULL AND :checkpoint_ns IS NULL) OR checkpoint_ns = :checkpoint_ns)
                """)
                await sess.execute(
                    update_sql,
                    {
                        "writes_json": new_json,
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                    },
                )
                await sess.commit()

    # ===== LIST (debug/ops)
    async def alist(self, config: RunnableConfig, *args, **kwargs):
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError("alist requires configurable.thread_id")
        checkpoint_ns = self._ns_from_config(config)

        async with self._session_factory() as sess:
            stmt = (
                select(CheckpointRow)
                .where(CheckpointRow.thread_id == thread_id)
                .where(CheckpointRow.checkpoint_ns == checkpoint_ns)
                .order_by(CheckpointRow.updated_at.desc())
            )
            res = await sess.execute(stmt)
            rows = res.scalars().all()
            for r in rows:
                yield {
                    "checkpoint": json.loads(r.checkpoint_blob) if r.checkpoint_blob else None,
                    "metadata":   json.loads(r.metadata_blob)   if r.metadata_blob else None,
                    "writes":     json.loads(r.writes_json)     if r.writes_json else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
