import io
import os
from typing import IO

from blake3 import blake3

DEFAULT_CHUNK = 8 * 1024 * 1024


def compute_blake3_hash(
    fp: str | IO[bytes],
    chunk_size: int = DEFAULT_CHUNK,
) -> str:
    if hasattr(fp, "read"):
        return _hash_file_obj(fp, chunk_size)

    with open(os.fspath(fp), "rb") as f:
        return _hash_file_obj(f, chunk_size)


def _hash_file_obj(file_obj: IO, chunk_size: int = DEFAULT_CHUNK) -> str:
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
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
        return h.hexdigest()
    finally:
        if seekable and orig_pos is not None:
            file_obj.seek(orig_pos)
