# async_sqlserver_checkpointer.py
import json
import uuid
from datetime import datetime
from typing import Any, Optional, Sequence, Tuple

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    select,
    update,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.runnables import RunnableConfig

Base = declarative_base()


class CheckpointRow(Base):
    __tablename__ = "checkpoints"
    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(255), index=True, nullable=False)
    checkpoint_id = Column(String(64), index=True, nullable=False)
    checkpoint_type = Column(String(255), nullable=True)
    checkpoint_blob = Column(Text, nullable=True)  # hex
    metadata_type = Column(String(255), nullable=True)
    metadata_blob = Column(Text, nullable=True)  # hex
    writes_json = Column(Text, nullable=True)  # JSON array of typed writes
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AsyncSQLServerCheckpointer(BaseCheckpointSaver):
    """
    Async SQLAlchemy backed checkpointer.
    - async_session_factory: callable that returns an AsyncSession (SQLAlchemy 1.4+ style).
    - serde: optional SerializerProtocol (default: JsonPlusSerializer(pickle_fallback=True))
    """

    def __init__(self, async_session_factory, *, serde: Optional[JsonPlusSerializer] = None):
        super().__init__(serde=serde)
        self._session_factory = async_session_factory
        self.jsonplus = serde or JsonPlusSerializer(pickle_fallback=True)

    # --------------------------
    # helper: hex/bytes conversions
    # --------------------------
    def _bytes_to_hex(self, b: Optional[bytes]) -> Optional[str]:
        return b.hex() if b is not None else None

    def _hex_to_bytes(self, h: Optional[str]) -> Optional[bytes]:
        return bytes.fromhex(h) if h else None

    # --------------------------
    # aget_tuple
    # --------------------------
    async def aget_tuple(self, config: RunnableConfig, *args, **kwargs) -> Optional[CheckpointTuple]:
        """Return the latest CheckpointTuple for the thread (or a specific checkpoint_id if provided)."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = (
            config["configurable"].get("checkpoint_id")
            or config["configurable"].get("thread_ts")
            or None
        )

        async with self._session_factory() as sess:  # type: AsyncSession
            stmt = select(CheckpointRow).where(CheckpointRow.thread_id == thread_id)
            if checkpoint_id:
                stmt = stmt.where(CheckpointRow.checkpoint_id == checkpoint_id)
            stmt = stmt.order_by(CheckpointRow.created_at.desc()).limit(1)
            res = await sess.execute(stmt)
            row = res.scalar_one_or_none()

            if not row:
                return None

            # deserialize checkpoint and metadata
            cp_type = row.checkpoint_type
            cp_bytes = self._hex_to_bytes(row.checkpoint_blob)
            checkpoint_obj = None
            if cp_type and cp_bytes is not None:
                checkpoint_obj = self.jsonplus.loads_typed((cp_type, cp_bytes))

            meta_type = row.metadata_type
            meta_bytes = self._hex_to_bytes(row.metadata_blob)
            metadata_obj = None
            if meta_type and meta_bytes is not None:
                metadata_obj = self.jsonplus.loads_typed((meta_type, meta_bytes))

            # Build a CheckpointTuple; parent_config left None (you can extend if you store parent)
            return CheckpointTuple(config=config, checkpoint=checkpoint_obj, parent_config=None)

    # --------------------------
    # aput (store full checkpoint)
    # signature must match langgraph: (config, checkpoint, metadata, new_versions) -> RunnableConfig
    # --------------------------
    async def aput(self, config: RunnableConfig, checkpoint: dict, metadata: dict, new_versions: dict, *args, **kwargs) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id") or str(uuid.uuid4())

        # typed dumps
        cp_type, cp_bytes = self.jsonplus.dumps_typed(checkpoint)
        meta_type, meta_bytes = self.jsonplus.dumps_typed(metadata or {})

        async with self._session_factory() as sess:
            row = CheckpointRow(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                checkpoint_type=cp_type,
                checkpoint_blob=self._bytes_to_hex(cp_bytes),
                metadata_type=meta_type,
                metadata_blob=self._bytes_to_hex(meta_bytes),
                writes_json=json.dumps([]),
                created_at=datetime.utcnow(),
            )
            sess.add(row)
            await sess.commit()

        # You may want to return an updated config (some official savers return runnable config
        # containing checkpoint_ns/checkpoint_id). Returning config as-is keeps it simple.
        return config

    # --------------------------
    # aput_writes (store pending writes)
    # signature must match: (config, writes, task_id, task_path: str = "") -> None
    # accept extra args for compatibility
    # --------------------------
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
        *args,
        **kwargs,
    ) -> None:
        """
        writes is a sequence of (channel, value) tuples.
        We'll append typed writes into the latest checkpoint row's writes_json array.
        """
        thread_id = config["configurable"]["thread_id"]

        async with self._session_factory() as sess:
            # get latest checkpoint row for the thread
            stmt = select(CheckpointRow).where(CheckpointRow.thread_id == thread_id).order_by(CheckpointRow.created_at.desc()).limit(1)
            res = await sess.execute(stmt)
            row = res.scalar_one_or_none()

            if not row:
                # if no checkpoint yet, create an empty one, then store writes
                checkpoint_id = config["configurable"].get("checkpoint_id") or str(uuid.uuid4())
                row = CheckpointRow(
                    thread_id=thread_id,
                    checkpoint_id=checkpoint_id,
                    checkpoint_type=None,
                    checkpoint_blob=None,
                    metadata_type=None,
                    metadata_blob=None,
                    writes_json=json.dumps([]),
                    created_at=datetime.utcnow(),
                )
                sess.add(row)
                await sess.flush()  # ensure row present

            existing = json.loads(row.writes_json or "[]")

            # append the new writes in typed form
            for idx, (channel, value) in enumerate(writes):
                v_type, v_bytes = self.jsonplus.dumps_typed(value)
                entry = {
                    "channel": channel,
                    "task_id": task_id,
                    "task_path": task_path,
                    "idx": idx,
                    "type": v_type,
                    "value_hex": self._bytes_to_hex(v_bytes),
                    "ts": datetime.utcnow().isoformat(),
                }
                existing.append(entry)

            # update row
            stmt = (
                update(CheckpointRow)
                .where(CheckpointRow.id == row.id)
                .values(writes_json=json.dumps(existing), created_at=datetime.utcnow())
            )
            await sess.execute(stmt)
            await sess.commit()

    # --------------------------
    # alist (optional but useful)
    # --------------------------
    async def alist(self, config: RunnableConfig, *args, **kwargs):
        thread_id = config["configurable"]["thread_id"]
        async with self._session_factory() as sess:
            stmt = select(CheckpointRow).where(CheckpointRow.thread_id == thread_id).order_by(CheckpointRow.created_at.desc())
            res = await sess.execute(stmt)
            for row in res.scalars().all():
                cp = None
                if row.checkpoint_type and row.checkpoint_blob:
                    cp = self.jsonplus.loads_typed((row.checkpoint_type, self._hex_to_bytes(row.checkpoint_blob)))
                yield CheckpointTuple(config=config, checkpoint=cp, parent_config=None)
