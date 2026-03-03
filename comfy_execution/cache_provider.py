"""
External Cache Provider API for distributed caching.

This module provides a public API for external cache providers, enabling
distributed caching across multiple ComfyUI instances (e.g., Kubernetes pods).

Public API is also available via:
    from comfy_api.latest import Caching

Example usage:
    from comfy_execution.cache_provider import (
        CacheProvider, CacheContext, CacheValue, register_cache_provider
    )

    class MyRedisProvider(CacheProvider):
        async def on_lookup(self, context: CacheContext) -> Optional[CacheValue]:
            # Check Redis/GCS for cached result
            ...

        async def on_store(self, context: CacheContext, value: CacheValue) -> None:
            # Store to Redis/GCS
            ...

    register_cache_provider(MyRedisProvider())
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, List
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import pickle
import threading

_logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class CacheContext:
    """Context passed to provider methods."""
    prompt_id: str          # Current prompt execution ID
    node_id: str            # Node being cached
    class_type: str         # Node class type (e.g., "KSampler")
    cache_key_hash: str     # SHA256 hex digest for external storage key


@dataclass
class CacheValue:
    """
    Value stored/retrieved from external cache.

    The ui field is optional - implementations may choose to skip it
    (e.g., if it contains non-portable data like local file paths).
    """
    outputs: list  # The tensor/value outputs
    ui: dict = None  # Optional UI data (may be skipped by implementations)


# ============================================================
# Provider Interface
# ============================================================

class CacheProvider(ABC):
    """
    Abstract base class for external cache providers.

    Async Safety:
        Provider methods are called from async context. Implementations
        can use async I/O (aiohttp, asyncpg, etc.) directly.

    Error Handling:
        All methods are wrapped in try/except by the caller. Exceptions
        are logged but never propagate to break execution.

    Performance Guidelines:
        - on_lookup: Should complete in <500ms (including network)
        - on_store: Fire-and-forget via asyncio.create_task
        - should_cache: Should be fast (<1ms), called frequently
    """

    @abstractmethod
    async def on_lookup(self, context: CacheContext) -> Optional[CacheValue]:
        """
        Check external storage for cached result.

        Called AFTER local cache miss (local-first for performance).

        Returns:
            CacheValue if found externally, None otherwise.

        Important:
            - Return None on any error (don't raise)
            - Validate data integrity before returning
        """
        pass

    @abstractmethod
    async def on_store(self, context: CacheContext, value: CacheValue) -> None:
        """
        Store value to external cache.

        Called AFTER value is stored in local cache.
        Dispatched as asyncio.create_task (fire-and-forget).

        Important:
            - Should never block execution
            - Handle serialization failures gracefully
        """
        pass

    def should_cache(self, context: CacheContext, value: Optional[CacheValue] = None) -> bool:
        """
        Filter which nodes should be externally cached.

        Called before on_lookup (value=None) and on_store (value provided).
        Return False to skip external caching for this node.

        Implementations can filter based on context.class_type, value size,
        or any custom logic. Use _estimate_value_size() to get value size.

        Default: Returns True (cache everything).
        """
        return True

    def on_prompt_start(self, prompt_id: str) -> None:
        """Called when prompt execution begins. Optional."""
        pass

    def on_prompt_end(self, prompt_id: str) -> None:
        """Called when prompt execution ends. Optional."""
        pass


# ============================================================
# Provider Registry
# ============================================================

_providers: List[CacheProvider] = []
_providers_lock = threading.Lock()
_providers_snapshot: Optional[Tuple[CacheProvider, ...]] = None


def register_cache_provider(provider: CacheProvider) -> None:
    """
    Register an external cache provider.

    Providers are called in registration order. First provider to return
    a result from on_lookup wins.
    """
    global _providers_snapshot
    with _providers_lock:
        if provider in _providers:
            _logger.warning(f"Provider {provider.__class__.__name__} already registered")
            return
        _providers.append(provider)
        _providers_snapshot = None  # Invalidate cache
        _logger.info(f"Registered cache provider: {provider.__class__.__name__}")


def unregister_cache_provider(provider: CacheProvider) -> None:
    """Remove a previously registered provider."""
    global _providers_snapshot
    with _providers_lock:
        try:
            _providers.remove(provider)
            _providers_snapshot = None
            _logger.info(f"Unregistered cache provider: {provider.__class__.__name__}")
        except ValueError:
            _logger.warning(f"Provider {provider.__class__.__name__} was not registered")


def _get_cache_providers() -> Tuple[CacheProvider, ...]:
    """Get registered providers (cached for performance). Internal."""
    global _providers_snapshot
    snapshot = _providers_snapshot
    if snapshot is not None:
        return snapshot
    with _providers_lock:
        if _providers_snapshot is not None:
            return _providers_snapshot
        _providers_snapshot = tuple(_providers)
        return _providers_snapshot


def _has_cache_providers() -> bool:
    """Fast check if any providers registered (no lock). Internal."""
    return bool(_providers)


def _clear_cache_providers() -> None:
    """Remove all providers. Useful for testing. Internal."""
    global _providers_snapshot
    with _providers_lock:
        _providers.clear()
        _providers_snapshot = None


# ============================================================
# Internal Utilities
# ============================================================

def _canonicalize(obj: Any) -> Any:
    """
    Convert an object to a canonical, JSON-serializable form.

    This ensures deterministic ordering regardless of Python's hash randomization,
    which is critical for cross-pod cache key consistency. Frozensets in particular
    have non-deterministic iteration order between Python sessions.
    """
    if isinstance(obj, frozenset):
        # Sort frozenset items for deterministic ordering
        return ("__frozenset__", sorted(
            [_canonicalize(item) for item in obj],
            key=lambda x: json.dumps(x, sort_keys=True)
        ))
    elif isinstance(obj, set):
        return ("__set__", sorted(
            [_canonicalize(item) for item in obj],
            key=lambda x: json.dumps(x, sort_keys=True)
        ))
    elif isinstance(obj, tuple):
        return ("__tuple__", [_canonicalize(item) for item in obj])
    elif isinstance(obj, list):
        return [_canonicalize(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, bytes):
        return ("__bytes__", obj.hex())
    elif hasattr(obj, 'value'):
        # Handle Unhashable class from ComfyUI
        return ("__unhashable__", _canonicalize(getattr(obj, 'value', None)))
    else:
        # For other types, use repr as fallback
        return ("__repr__", repr(obj))


def _serialize_cache_key(cache_key: Any) -> Optional[str]:
    """
    Serialize cache key to a hex digest string for external storage.

    Returns SHA256 hex string suitable for Redis/database keys,
    or None if serialization fails entirely (fail-closed).

    Note: Uses canonicalize + JSON serialization instead of pickle because
    pickle is NOT deterministic across Python sessions due to hash randomization
    affecting frozenset iteration order. This is critical for distributed caching
    where different pods need to compute the same hash for identical inputs.
    """
    try:
        canonical = _canonicalize(cache_key)
        json_str = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    except Exception as e:
        _logger.warning(f"Failed to serialize cache key: {e}")
        # Fallback to pickle (non-deterministic but better than nothing)
        try:
            serialized = pickle.dumps(cache_key, protocol=4)
            return hashlib.sha256(serialized).hexdigest()
        except Exception as fallback_error:
            _logger.warning(f"Failed pickle fallback for cache key: {fallback_error}")
            return None


def _contains_nan(obj: Any) -> bool:
    """
    Check if cache key contains NaN (indicates uncacheable node).

    NaN != NaN in Python, so local cache never hits. But serialized
    NaN would match, causing incorrect external hits. Must skip these.
    """
    if isinstance(obj, float):
        try:
            return math.isnan(obj)
        except (TypeError, ValueError):
            return False
    if hasattr(obj, 'value'):  # Unhashable class
        val = getattr(obj, 'value', None)
        if isinstance(val, float):
            try:
                return math.isnan(val)
            except (TypeError, ValueError):
                return False
    if isinstance(obj, (frozenset, tuple, list, set)):
        return any(_contains_nan(item) for item in obj)
    if isinstance(obj, dict):
        return any(_contains_nan(k) or _contains_nan(v) for k, v in obj.items())
    return False


def _estimate_value_size(value: CacheValue) -> int:
    """Estimate serialized size in bytes. Useful for size-based filtering. Internal."""
    try:
        import torch
    except ImportError:
        return 0

    total = 0

    def estimate(obj):
        nonlocal total
        if isinstance(obj, torch.Tensor):
            total += obj.numel() * obj.element_size()
        elif isinstance(obj, dict):
            for v in obj.values():
                estimate(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                estimate(item)

    for output in value.outputs:
        estimate(output)
    return total
