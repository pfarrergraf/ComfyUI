import io
import os
from dataclasses import dataclass
from typing import IO, Any, Callable

from blake3 import blake3

DEFAULT_CHUNK = 8 * 1024 * 1024

InterruptCheck = Callable[[], bool]


@dataclass
class HashCheckpoint:
    """Saved state for resuming an interrupted hash computation."""

    bytes_processed: int
    hasher: Any  # blake3 hasher instance
    mtime_ns: int = 0
    file_size: int = 0


def compute_blake3_hash(
    fp: str | IO[bytes],
    chunk_size: int = DEFAULT_CHUNK,
    interrupt_check: InterruptCheck | None = None,
    checkpoint: HashCheckpoint | None = None,
) -> tuple[str | None, HashCheckpoint | None]:
    """Compute BLAKE3 hash of a file, with optional checkpoint support.

    Args:
        fp: File path or file-like object
        chunk_size: Size of chunks to read at a time
        interrupt_check: Optional callable that returns True if the operation
            should be interrupted (e.g. paused or cancelled). Must be
            non-blocking so file handles are released immediately. Checked
            between chunk reads.
        checkpoint: Optional checkpoint to resume from (file paths only)

    Returns:
        Tuple of (hex_digest, None) on completion, or
        (None, checkpoint) on interruption (file paths only), or
        (None, None) on interruption of a file object
    """
    if hasattr(fp, "read"):
        digest = _hash_file_obj(fp, chunk_size, interrupt_check)
        return digest, None

    with open(os.fspath(fp), "rb") as f:
        if checkpoint is not None:
            f.seek(checkpoint.bytes_processed)
            h = checkpoint.hasher
            bytes_processed = checkpoint.bytes_processed
        else:
            h = blake3()
            bytes_processed = 0

        if chunk_size <= 0:
            chunk_size = DEFAULT_CHUNK

        while True:
            if interrupt_check is not None and interrupt_check():
                return None, HashCheckpoint(
                    bytes_processed=bytes_processed,
                    hasher=h,
                )
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            bytes_processed += len(chunk)

        return h.hexdigest(), None


def _hash_file_obj(
    file_obj: IO,
    chunk_size: int = DEFAULT_CHUNK,
    interrupt_check: InterruptCheck | None = None,
) -> str | None:
    if chunk_size <= 0:
        chunk_size = DEFAULT_CHUNK

    seekable = getattr(file_obj, "seekable", lambda: False)()
    orig_pos = None

    if seekable:
        try:
            orig_pos = file_obj.tell()
            if orig_pos != 0:
                file_obj.seek(0)
        except io.UnsupportedOperation:
            seekable = False
            orig_pos = None

    try:
        h = blake3()
        while True:
            if interrupt_check is not None and interrupt_check():
                return None
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
        return h.hexdigest()
    finally:
        if seekable and orig_pos is not None:
            file_obj.seek(orig_pos)
