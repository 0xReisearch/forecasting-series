#!/usr/bin/env python3
"""Adapt-1 Reference REST Client (Python 3 standard library only).

Provides a clean, zero-dependency HTTP client implementing Rei Labs' production
wire hygiene rules for the Adapt-1 managed stateful API:

  - HTTP Bearer-token auth via ADAPT_API_BASE and ADAPT_API_KEY env vars.
  - Non-default User-Agent header (avoids edge gateway HTTP 403 blocks).
  - Strict RFC-8259 JSON serialization (allow_nan=False; bool checked before int).
  - Idempotency-safe exponential-backoff retries:
      * Idempotent reads/DELETE/explain/state retry (429, 500, 502, 503, 504)
        and transient socket/read timeouts.
      * State-mutating POSTs retry ONLY (429, 502, 503, 504) and fail fast on
        4xx and 500 to avoid duplicate state mutations.
  - Guaranteed try/finally domain lifecycle context manager with verified
    DELETE -> GET 404 teardown.
  - Automatic <= 50 operation chunking for POST /batch and POST /value/observe.

Usage:
    export ADAPT_API_BASE="https://rei-neuroadapt-api.reilabs.org/api/v1"
    export ADAPT_API_KEY="<your-api-key>"

    # Read-only connectivity check (lists active domains; no state mutation)
    python3 scripts/adapt1_client.py --check

    # Offline self-test of strict RFC-8259 sanitization and chunking helpers
    python3 scripts/adapt1_client.py --self-test
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import random
import socket
import sys
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_USER_AGENT = "adapt1-replication-client/1.0"
DEFAULT_TIMEOUT_SECONDS = 90.0
READ_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
WRITE_RETRY_STATUSES = frozenset({429, 502, 503, 504})
MAX_BATCH_OPS = 50
MAX_VALUE_OBSERVE_ITEMS = 50


class Adapt1APIError(RuntimeError):
    """Raised when the Adapt-1 API returns a non-success HTTP status."""

    def __init__(self, status: int, method: str, path: str, body: str) -> None:
        self.status = status
        self.method = method
        self.path = path
        self.body = body
        snippet = body[:400].replace("\n", " ")
        super().__init__(f"HTTP {status} on {method} {path}: {snippet}")


def sanitize_json(obj: Any) -> Any:
    """Recursively convert values into strict RFC-8259 JSON-safe primitives.

    Non-finite floats (NaN, +Inf, -Inf) are mapped to None so json.dumps with
    allow_nan=False never emits invalid JSON tokens. bool is checked strictly
    before int because bool subclasses int in Python.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if hasattr(obj, "item") and callable(obj.item):
        return sanitize_json(obj.item())
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    raise TypeError(f"Unsupported JSON payload type: {type(obj).__name__}")


def encode_json(payload: Dict[str, Any]) -> bytes:
    """Encode a payload dictionary as compact, strict RFC-8259 UTF-8 JSON."""
    cleaned = sanitize_json(payload)
    return json.dumps(cleaned, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _reject_non_rfc_constant(token: str) -> None:
    raise ValueError(f"Non-RFC-8259 JSON constant in response: {token}")


def assert_complete_finite_values(values: Dict[str, Any]) -> None:
    """Client-side guard for POST /events and POST /batch event values.

    Adapt-1 silently drops event records with missing or null input/target
    values (accepted: false). Reject None or non-finite floats before sending.
    """
    for key, val in values.items():
        if val is None:
            raise ValueError(f"Event value for {key!r} is None (must be complete and finite)")
        if isinstance(val, float) and not math.isfinite(val):
            raise ValueError(f"Event value for {key!r} is non-finite: {val}")


def assert_lossless_futures(response: Dict[str, Any], tol: float = 1e-5) -> List[Dict[str, Any]]:
    """Verify lossless joint branch export on a /query response and return branches."""
    tp = response.get("transition_prediction") or {}
    futures = tp.get("futures") or {}
    status = futures.get("status")
    if status != "predicted":
        raise ValueError(f"Expected futures.status == 'predicted', got {status!r}")
    omitted = float(futures.get("omitted_probability", 0.0))
    if omitted >= 1e-12:
        raise ValueError(
            f"Lossy futures export detected (omitted_probability={omitted:.3e}); "
            "keep max_samples == max_support <= 128 and max_branches == 128 for lossless joint tails."
        )
    branches = futures.get("branches") or []
    total_prob = sum(float(b["probability"]) for b in branches)
    if abs(1.0 - total_prob) > tol:
        raise ValueError(f"Branch probabilities sum to {total_prob:.8f}, outside 1 +/- {tol}")
    return branches


def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield fixed-size slices of a sequence."""
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


class Adapt1Client:
    """Zero-dependency REST client for the Adapt-1 stateful API."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 4,
    ) -> None:
        base = api_base if api_base is not None else os.environ.get("ADAPT_API_BASE", "")
        key = api_key if api_key is not None else os.environ.get("ADAPT_API_KEY", "")
        if not base:
            raise ValueError("ADAPT_API_BASE is not set (expected https://rei-neuroadapt-api.reilabs.org/api/v1)")
        if not key:
            raise ValueError("ADAPT_API_KEY is not set")
        self.api_base = base.rstrip("/")
        self.api_key = key
        self.user_agent = user_agent
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        params: Optional[Dict[str, str]] = None,
        idempotent: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute an HTTP request with idempotency-safe retry handling."""
        method_up = method.upper()
        is_idempotent = (method_up in {"GET", "HEAD", "DELETE"}) if idempotent is None else bool(idempotent)
        retry_statuses = READ_RETRY_STATUSES if is_idempotent else WRITE_RETRY_STATUSES

        url = f"{self.api_base}{path}"
        if params:
            query_str = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            if query_str:
                url = f"{url}?{query_str}"

        body_bytes = encode_json(payload) if payload is not None else None
        req_timeout = self.timeout if timeout is None else float(timeout)

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers=self._headers,
                method=method_up,
            )
            try:
                with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    if not raw.strip():
                        return {}
                    return json.loads(raw, parse_constant=_reject_non_rfc_constant)
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")
                if exc.code in retry_statuses and attempt < self.max_retries:
                    delay = min(16.0, (0.5 * (2**attempt)) + random.uniform(0.0, 0.35))
                    time.sleep(delay)
                    last_exc = Adapt1APIError(exc.code, method_up, path, err_body)
                    continue
                raise Adapt1APIError(exc.code, method_up, path, err_body) from exc
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                # Never retry DNS resolution errors; retry transient timeouts only on idempotent calls.
                reason = getattr(exc, "reason", exc)
                is_dns = isinstance(reason, socket.gaierror)
                if is_idempotent and not is_dns and attempt < self.max_retries:
                    delay = min(16.0, (0.5 * (2**attempt)) + random.uniform(0.0, 0.35))
                    time.sleep(delay)
                    last_exc = exc
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    # --- Domain lifecycle ---

    def list_domains(self, session_id: Optional[str] = None) -> Any:
        params = {"session_id": session_id} if session_id else None
        return self.request("GET", "/domains", params=params, idempotent=True)

    def create_domain(self, declaration: Dict[str, Any]) -> Any:
        return self.request("POST", "/domains", payload=declaration, idempotent=False)

    def get_domain(self, domain_id: str, session_id: Optional[str] = None) -> Any:
        params = {"session_id": session_id} if session_id else None
        return self.request("GET", f"/domains/{domain_id}", params=params, idempotent=True)

    def delete_domain(
        self,
        domain_id: str,
        session_id: Optional[str] = None,
        verify_404: bool = True,
    ) -> None:
        """Delete a domain and optionally assert GET returns HTTP 404."""
        params = {"session_id": session_id} if session_id else None
        try:
            self.request("DELETE", f"/domains/{domain_id}", params=params, idempotent=True)
        except Adapt1APIError as exc:
            if exc.status != 404:
                raise
        if verify_404:
            for attempt in range(4):
                try:
                    self.get_domain(domain_id, session_id=session_id)
                except Adapt1APIError as exc:
                    if exc.status == 404:
                        return
                    raise
                time.sleep(0.25 * (attempt + 1))
            raise RuntimeError(f"Domain {domain_id!r} still returned HTTP 200 after DELETE")

    @contextmanager
    def domain_session(self, declaration: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Create a domain and guarantee verified DELETE -> GET 404 teardown."""
        created = self.create_domain(declaration)
        domain_id = str(created.get("domain_id") or declaration["domain_id"])
        session_id = declaration.get("session_id")
        try:
            yield created
        finally:
            self.delete_domain(domain_id, session_id=session_id, verify_404=True)

    # --- Forecasting, policy, sequential & explanation surfaces ---

    def query(self, domain_id: str, payload: Dict[str, Any]) -> Any:
        return self.request("POST", f"/domains/{domain_id}/query", payload=payload, idempotent=False)

    def post_event(self, domain_id: str, payload: Dict[str, Any]) -> Any:
        if "values" in payload and isinstance(payload["values"], dict):
            assert_complete_finite_values(payload["values"])
        return self.request("POST", f"/domains/{domain_id}/events", payload=payload, idempotent=False)

    def post_feedback(self, domain_id: str, payload: Dict[str, Any]) -> Any:
        return self.request("POST", f"/domains/{domain_id}/feedback", payload=payload, idempotent=False)

    def post_batch(
        self,
        domain_id: str,
        session_id: str,
        operations: Sequence[Dict[str, Any]],
        chunk_size: int = MAX_BATCH_OPS,
    ) -> List[Any]:
        """Submit event or feedback operations in <= 50-op chunks."""
        if chunk_size > MAX_BATCH_OPS:
            raise ValueError(f"POST /batch accepts at most {MAX_BATCH_OPS} operations per request")
        responses = []
        for batch in chunked(operations, chunk_size):
            for op in batch:
                if op.get("operation") == "event" and isinstance(op.get("values"), dict):
                    assert_complete_finite_values(op["values"])
            resp = self.request(
                "POST",
                f"/domains/{domain_id}/batch",
                payload={"session_id": session_id, "operations": list(batch)},
                idempotent=False,
                timeout=max(self.timeout, 120.0),
            )
            responses.append(resp)
        return responses

    def explain(self, domain_id: str, payload: Dict[str, Any]) -> Any:
        return self.request("POST", f"/domains/{domain_id}/explain", payload=payload, idempotent=True)

    def policy_reset(self, domain_id: str, session_id: str) -> Any:
        return self.request(
            "POST",
            f"/domains/{domain_id}/policy/reset",
            payload={"session_id": session_id},
            idempotent=False,
        )

    # --- Stateful /value/* lifecycle ---

    def value_call(self, domain_id: str, endpoint: str, payload: Dict[str, Any]) -> Any:
        allowed = {"configure", "observe", "train", "predict", "promote", "freeze", "reset", "state"}
        if endpoint not in allowed:
            raise ValueError(f"Unknown /value endpoint {endpoint!r}; expected one of {sorted(allowed)}")
        is_idempotent = endpoint in {"state", "predict"}
        return self.request(
            "POST",
            f"/domains/{domain_id}/value/{endpoint}",
            payload=payload,
            idempotent=is_idempotent,
        )

    def value_observe_chunked(
        self,
        domain_id: str,
        session_id: str,
        observations: Sequence[Dict[str, Any]],
        chunk_size: int = MAX_VALUE_OBSERVE_ITEMS,
    ) -> List[Any]:
        """Upload /value/observe records in <= 50-item chunks (API hard cap)."""
        if chunk_size > MAX_VALUE_OBSERVE_ITEMS:
            raise ValueError(f"POST /value/observe accepts at most {MAX_VALUE_OBSERVE_ITEMS} observations per request")
        responses = []
        for batch in chunked(observations, chunk_size):
            responses.append(
                self.value_call(
                    domain_id,
                    "observe",
                    {"session_id": session_id, "observations": list(batch)},
                )
            )
        return responses


def _run_offline_self_test() -> int:
    sample = {
        "flag_true": True,
        "flag_false": False,
        "count": 42,
        "finite": 1.25,
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "nested": [1, True, float("-inf"), {"x": 3.5}],
    }
    encoded = encode_json(sample).decode("utf-8")
    parsed = json.loads(encoded, parse_constant=_reject_non_rfc_constant)
    assert parsed["flag_true"] is True and type(parsed["flag_true"]) is bool
    assert parsed["flag_false"] is False and type(parsed["flag_false"]) is bool
    assert parsed["nan_val"] is None and parsed["inf_val"] is None
    assert parsed["nested"][1] is True and parsed["nested"][2] is None
    batches = list(chunked(list(range(123)), MAX_BATCH_OPS))
    assert [len(b) for b in batches] == [50, 50, 23]
    print("PASS: adapt1_client offline RFC-8259 & chunking self-test passed.")
    return 0


def _run_connectivity_check(session_id: Optional[str]) -> int:
    try:
        client = Adapt1Client()
        res = client.list_domains(session_id=session_id)
    except Exception as exc:
        print(f"FAIL: Adapt-1 connectivity check failed: {exc}", file=sys.stderr)
        return 1
    domains = res.get("domains", res) if isinstance(res, dict) else res
    n_domains = len(domains) if isinstance(domains, list) else "unknown"
    print(f"PASS: Connected to Adapt-1 API (active domains visible: {n_domains}).")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Adapt-1 reference REST client CLI")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a read-only GET /domains connectivity check using ADAPT_API_BASE and ADAPT_API_KEY.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session_id filter for --check.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline RFC-8259 serializer and chunking unit assertions (no network required).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.self_test:
        return _run_offline_self_test()
    if args.check:
        return _run_connectivity_check(args.session_id)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
