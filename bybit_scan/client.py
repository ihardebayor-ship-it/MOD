"""Thin, rate-limited client for Bybit V5 *public* market endpoints.

No API key is needed for anything in this module. Responses are cached on
disk so re-running a scan does not re-download everything.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests

MAINNET = "https://api.bybit.com"
TESTNET = "https://api-testnet.bybit.com"

CATEGORIES = ("spot", "linear", "inverse", "option")


class BybitError(RuntimeError):
    pass


class RateLimiter:
    """Simple token bucket: at most `rate` calls per second across threads."""

    def __init__(self, rate: float):
        self.interval = 1.0 / rate
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.interval


class BybitPublic:
    def __init__(
        self,
        base_url: str = MAINNET,
        cache_dir: str | Path | None = ".cache/bybit",
        rate: float = 8.0,
        cache_ttl: float = 6 * 3600,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl
        self.limiter = RateLimiter(rate)
        self.session = session or requests.Session()
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- low level ---------------------------------------------------------
    def _cache_path(self, path: str, params: dict[str, Any]) -> Path | None:
        if not self.cache_dir:
            return None
        key = json.dumps([path, sorted(params.items())], separators=(",", ":"))
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".json")

    def get(self, path: str, params: dict[str, Any] | None = None, *, cache: bool = True) -> dict[str, Any]:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        cpath = self._cache_path(path, params) if cache else None
        if cpath and cpath.exists() and time.time() - cpath.stat().st_mtime < self.cache_ttl:
            return json.loads(cpath.read_text())

        last_err: Exception | None = None
        for attempt in range(5):
            self.limiter.wait()
            try:
                r = self.session.get(self.base_url + path, params=params, timeout=30)
                if r.status_code == 403:
                    raise BybitError(
                        "HTTP 403 from Bybit. This is usually a geo-block: Bybit refuses "
                        "requests from the US and some other regions. Run from a permitted "
                        "country or a VPS in one."
                    )
                if r.status_code in (429, 500, 502, 503, 504):
                    raise BybitError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
            except BybitError as e:
                if "403" in str(e):
                    raise
                last_err = e
            except (requests.RequestException, ValueError) as e:
                last_err = e
            else:
                if data.get("retCode") != 0:
                    raise BybitError(f"{path} {params}: retCode={data.get('retCode')} {data.get('retMsg')}")
                if cpath:
                    cpath.write_text(json.dumps(data))
                return data
            time.sleep(min(2**attempt, 16))
        raise BybitError(f"giving up on {path} {params}: {last_err}")

    # -- paginated helpers ---------------------------------------------------
    def _paged(self, path: str, params: dict[str, Any], limit: int = 1000) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            data = self.get(path, {**params, "limit": limit, "cursor": cursor})
            result = data["result"]
            out.extend(result.get("list", []))
            cursor = result.get("nextPageCursor") or None
            if not cursor:
                return out

    # -- public endpoints ----------------------------------------------------
    def server_time(self) -> int:
        return int(self.get("/v5/market/time", cache=False)["result"]["timeSecond"])

    def instruments(self, category: str, status: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"category": category, "status": status}
        if category == "option":
            params["baseCoin"] = "All"
        return self._paged("/v5/market/instruments-info", params)

    def tickers(self, category: str, base_coin: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"category": category, "baseCoin": base_coin}
        return self.get("/v5/market/tickers", params, cache=False)["result"]["list"]

    def kline(
        self,
        category: str,
        symbol: str,
        interval: str = "D",
        start: int | None = None,
        end: int | None = None,
        limit: int = 1000,
    ) -> list[list[str]]:
        params = {"category": category, "symbol": symbol, "interval": interval, "start": start, "end": end, "limit": limit}
        return self.get("/v5/market/kline", params)["result"]["list"]

    def funding_history(self, category: str, symbol: str, start: int | None = None, end: int | None = None) -> list[dict[str, Any]]:
        params = {"category": category, "symbol": symbol, "startTime": start, "endTime": end, "limit": 200}
        return self.get("/v5/market/funding/history", params)["result"]["list"]
