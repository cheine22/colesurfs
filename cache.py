"""
colesurfs — Thread-safe TTL cache utility + API call counter.

Entries are written through to .cache/*.json and restored on startup so
slow fetchers (CMEMS) survive a process restart.
"""
import hashlib
import re
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps


_DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


class _TTLStore:
    def __init__(self):
        self._data: dict = {}
        self._lock = threading.Lock()
        self._prefix_ttls: dict[str, int] = {}
        self._restore_from_disk()

    def register_prefix_ttl(self, prefix: str, ttl: int):
        """Called at decorator-definition time to record the configured TTL per key prefix."""
        existing = self._prefix_ttls.get(prefix)
        self._prefix_ttls[prefix] = min(existing, ttl) if existing is not None else ttl

    def get(self, key):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None, False
            value, ts, ttl = entry
            configured = self._prefix_ttls.get(key.split(':')[0])
            effective_ttl = min(ttl, configured) if configured is not None else ttl
            if time.monotonic() - ts < effective_ttl:
                return value, True
            del self._data[key]
            return None, False

    def set(self, key, value, ttl):
        with self._lock:
            self._data[key] = (value, time.monotonic(), ttl)
        # Write-through to disk (fire-and-forget, don't block the caller)
        try:
            self._write_disk(key, value, ttl)
        except Exception:
            pass

    def clear(self):
        with self._lock:
            self._data.clear()
        # Also clear disk cache
        try:
            if os.path.isdir(_DISK_CACHE_DIR):
                for f in os.listdir(_DISK_CACHE_DIR):
                    # only our md5-named entries; .cache/ also holds lkg_forecast.json
                    if re.fullmatch(r"[0-9a-f]{32}\.json", f):
                        os.unlink(os.path.join(_DISK_CACHE_DIR, f))
        except Exception:
            pass

    def get_age(self, key) -> float | None:
        """Return seconds since this key was cached, or None if missing/expired."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            _, ts, ttl = entry
            age = time.monotonic() - ts
            if age >= ttl:
                return None
            return age

    # ─── Disk persistence helpers ────────────────────────────────────────────

    @staticmethod
    def _disk_path(key: str) -> str:
        safe = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(_DISK_CACHE_DIR, f"{safe}.json")

    def _write_disk(self, key: str, value, ttl: int):
        """Write a cache entry to disk as JSON."""
        os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
        path = self._disk_path(key)
        try:
            with open(path, "w") as f:
                json.dump({"key": key, "value": value, "ttl": ttl,
                           "wall_ts": time.time()}, f, separators=(',', ':'))
        except (TypeError, ValueError):
            # Value not JSON-serializable — skip disk cache for this entry
            pass

    def _restore_from_disk(self):
        """Restore cache entries from disk on startup. Skips expired entries."""
        if not os.path.isdir(_DISK_CACHE_DIR):
            return
        now = time.time()
        restored = 0
        for fname in os.listdir(_DISK_CACHE_DIR):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(_DISK_CACHE_DIR, fname)
            try:
                with open(path) as f:
                    entry = json.load(f)
                age = now - entry["wall_ts"]
                if age < entry["ttl"]:
                    # Reconstruct monotonic timestamp
                    mono_ts = time.monotonic() - age
                    self._data[entry["key"]] = (entry["value"], mono_ts, entry["ttl"])
                    restored += 1
                else:
                    # Expired on disk — clean up
                    os.unlink(path)
            except Exception:
                pass
        if restored:
            print(f"[cache] restored {restored} entries from disk")


_store = _TTLStore()

# Single-flight: one lock per cache key so concurrent requests on an
# expired entry don't stampede the upstream (a cold CMEMS fetch can take
# 90 s — without this, every waiting request fired its own copy).
_keylocks_guard = threading.Lock()
_keylocks: dict[str, threading.Lock] = {}


def _key_lock(key: str) -> threading.Lock:
    with _keylocks_guard:
        lk = _keylocks.get(key)
        if lk is None:
            lk = _keylocks[key] = threading.Lock()
        return lk


# ─── API call counter (resets daily at midnight UTC) ─────────────────────────
_api_lock = threading.Lock()
_api_day: str = ""           # "YYYY-MM-DD" of current counter
_api_counts: dict[str, int] = {}   # {label: count}


def record_api_calls(label: str, n_points: int = 1):
    """Increment the daily API call counter for a given label."""
    global _api_day, _api_counts
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = str(label)   # a non-str label mixes key types and breaks jsonify's
                       # sort_keys in /api/status — 500s the whole endpoint
    with _api_lock:
        if today != _api_day:
            _api_day = today
            _api_counts = {}
        _api_counts[key] = _api_counts.get(key, 0) + n_points


def get_api_usage() -> dict:
    """Return {day, counts: {label: n}, total}."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _api_lock:
        if today != _api_day:
            return {"day": today, "counts": {}, "total": 0}
        return {
            "day": _api_day,
            "counts": dict(_api_counts),
            "total": sum(_api_counts.values()),
        }


def ttl_cache(ttl_seconds: int = 3600, skip_none: bool = False):
    """
    Decorator: cache function return value for `ttl_seconds`.
    Cache key = function name + stringified positional args.
    skip_none=True: do not cache None results so failures are retried on the next call.
    """
    def decorator(func):
        prefix = f"{func.__module__}.{func.__qualname__}"
        _store.register_prefix_ttl(prefix, ttl_seconds)
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{prefix}:{args}:{sorted(kwargs.items())}"
            cached, hit = _store.get(key)
            if hit:
                return cached
            with _key_lock(key):
                cached, hit = _store.get(key)
                if hit:
                    return cached
                result = func(*args, **kwargs)
                if result is not None or not skip_none:
                    _store.set(key, result, ttl_seconds)
                return result
        return wrapper
    return decorator


def get_cache_age(key: str) -> float | None:
    """Return age in seconds for a cache key, or None if not cached."""
    return _store.get_age(key)


def age_of(func, *args, **kwargs) -> float | None:
    """Age of the entry a decorated fetcher would serve for these args."""
    return _store.get_age(f"{func.__module__}.{func.__qualname__}:{args}:{sorted(kwargs.items())}")


# A due re-fetch runs on a background pool, one per key, and every caller
# shares it. Callers wait for it by default — the warmer, the widgets and the
# fetchers that cache the result again themselves (/gland, /csc) must get the
# new run, not the one it replaces. A caller holding the previous run can cap
# the wait with refresh_wait(): the dashboard's EURO request waits at most
# REFRESH_WAIT_S, then takes the cached run while the fetch finishes. A failed
# re-fetch is not retried for REFRESH_BACKOFF_S; the cached run is served
# meanwhile (nothing once the hard TTL lapses — callers fall back to their
# last-known-good). clear_all() forgets the backoff along with the values.
# Without these, a Copernicus login outage (2026-10-13) queued every EURO
# request for ~25 min behind failing re-fetches and the page's 9 s fetch
# timeout read it as a forecast outage.
REFRESH_WAIT_S = 6.0
REFRESH_BACKOFF_S = 900
_refresh_pool = ThreadPoolExecutor(max_workers=12, thread_name_prefix="refresh")
_refresh_guard = threading.Lock()
_refresh_inflight: dict = {}   # key → Future of the running re-fetch
_refresh_failed: dict = {}     # key → monotonic time of the last failed re-fetch
_refresh_tls = threading.local()


@contextmanager
def refresh_wait(seconds):
    """On this thread, inside the block, a model_aware_cache call that already
    holds a cached value waits at most `seconds` for a due re-fetch and then
    returns the cached value. None (the default outside any block) waits."""
    prev = getattr(_refresh_tls, "wait", None)
    _refresh_tls.wait = seconds
    try:
        yield
    finally:
        _refresh_tls.wait = prev


def model_aware_cache(hard_ttl: int = 21600, model_arg_index: int = 0,
                      quiet: bool = False):
    """
    Decorator: cache with model-run-aware invalidation.
    Unlike ttl_cache, this uses a long hard TTL (default 6h) but allows
    early invalidation when a new model run becomes available.

    The decorated function's first positional arg (by default) is model_key.
    Requires a `new_run_checker` to be set on the module — a callable
    (model_key, cache_age_sec) -> bool.

    Flow:
      1. If cached and within hard TTL:
         a. Ask new_run_checker if a new model run is available since cache time
         b. If no new run → return cached value (skip fetch)
         c. If new run → re-fetch in the background and wait for it; inside
            refresh_wait(s), at most s seconds, then the cached value
      2. If not cached or expired beyond hard TTL → fetch and wait for it
      3. Within REFRESH_BACKOFF_S of a failed fetch → no fetch; the cached
         value, or None
    """
    def decorator(func):
        prefix = f"{func.__module__}.{func.__qualname__}"
        _store.register_prefix_ttl(prefix, hard_ttl)
        def _fetch(key, args, kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                print(f"[smart-cache] {func.__qualname__}: {type(e).__name__}: {e}")
                result = None
            if result is not None:
                _store.set(key, result, hard_ttl)
                _refresh_failed.pop(key, None)
            else:
                _refresh_failed[key] = time.monotonic()
            return result

        def _start(key, args, kwargs, model_key, age):
            """The running fetch for key — single-flight — started if none is."""
            with _refresh_guard:
                fut = _refresh_inflight.get(key)
                if fut is None or fut.done():
                    if age is not None:
                        print(f"[smart-cache] {func.__qualname__}({model_key}): "
                              f"cached {age:.0f}s ago, new run available → re-fetching")
                    fut = _refresh_inflight[key] = _refresh_pool.submit(_fetch, key, args, kwargs)
                return fut

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{prefix}:{args}:{sorted(kwargs.items())}"
            cached, hit = _store.get(key)
            age = _store.get_age(key) if hit else None
            model_key = args[model_arg_index] if len(args) > model_arg_index else "EURO"
            checker = getattr(wrapper, '_new_run_checker', None)
            if hit and checker and age is not None and not checker(model_key, age):
                if not quiet:
                    print(f"[smart-cache] {func.__qualname__}({model_key}): "
                          f"cached {age:.0f}s ago, no new run → skip fetch")
                return cached
            failed = _refresh_failed.get(key)
            if failed is not None and time.monotonic() - failed < REFRESH_BACKOFF_S:
                return cached if hit else None
            fut = _start(key, args, kwargs, model_key, age)
            try:
                result = fut.result(timeout=getattr(_refresh_tls, "wait", None) if hit else None)
            except _FutureTimeout:
                return cached
            return result if result is not None else (cached if hit else None)

        wrapper._new_run_checker = None  # set by caller
        return wrapper
    return decorator


def clear_all():
    _store.clear()
    _refresh_failed.clear()   # a refresh after an outage must try the upstream again
