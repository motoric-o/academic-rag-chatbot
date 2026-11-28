"""Token bucket rate limiter and persistent daily quota tracker.

Owner: Member C (Francisco Valentino; see docs/project_spec.md §10 & §13).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class RateLimitExceededError(Exception):
    """Raised when daily quota (RPD or TPD) is completely exhausted."""
    pass


class PersistentRateLimiter:
    """Manages RPM, TPM, RPD, and TPD quotas with state persistence and backoff."""

    def __init__(
        self,
        rpm: int = 30,
        tpm: int = 8000,
        rpd: int = 1000,
        tpd: int = 200000,
        state_path: str | Path = "data/cache/rate_limiter_state.json",
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.tpd = tpd
        self.state_path = Path(state_path)
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn

        # Per-minute token buckets
        self.req_capacity = float(rpm)
        self.token_capacity = float(tpm)
        self.req_tokens = float(rpm)
        self.token_tokens = float(tpm)
        self.req_fill_rate = float(rpm) / 60.0  # requests per second
        self.token_fill_rate = float(tpm) / 60.0  # tokens per second
        self.last_leak_time = self.time_fn()

        # Load persistent daily state
        self.current_date = self._get_today_str()
        self.daily_requests_used = 0
        self.daily_tokens_used = 0
        self._load_state()

    def _get_today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load_state(self) -> None:
        if self.state_path.is_file():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("date") == self.current_date:
                    self.daily_requests_used = data.get("daily_requests_used", 0)
                    self.daily_tokens_used = data.get("daily_tokens_used", 0)
            except Exception:
                pass

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "date": self.current_date,
                        "daily_requests_used": self.daily_requests_used,
                        "daily_tokens_used": self.daily_tokens_used,
                        "last_updated": self.time_fn(),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    def _check_and_reset_day(self) -> None:
        today = self._get_today_str()
        if today != self.current_date:
            self.current_date = today
            self.daily_requests_used = 0
            self.daily_tokens_used = 0
            self._save_state()

    def _refill_buckets(self) -> None:
        now = self.time_fn()
        elapsed = max(0.0, now - self.last_leak_time)
        self.last_leak_time = now

        self.req_tokens = min(self.req_capacity, self.req_tokens + elapsed * self.req_fill_rate)
        self.token_tokens = min(self.token_capacity, self.token_tokens + elapsed * self.token_fill_rate)

    def acquire(self, estimated_tokens: int = 2500) -> None:
        """Wait until sufficient capacity exists under RPM, TPM, RPD, and TPD."""
        self._check_and_reset_day()

        if self.daily_requests_used + 1 > self.rpd:
            raise RateLimitExceededError(
                f"Daily request quota (RPD={self.rpd}) exhausted for {self.current_date}."
            )
        if self.daily_tokens_used + estimated_tokens > self.tpd:
            raise RateLimitExceededError(
                f"Daily token quota (TPD={self.tpd}) exhausted for {self.current_date}."
            )

        while True:
            self._refill_buckets()
            needed_wait = 0.0

            if self.req_tokens < 1.0:
                deficit = 1.0 - self.req_tokens
                wait_req = deficit / self.req_fill_rate
                needed_wait = max(needed_wait, wait_req)

            if self.token_tokens < float(estimated_tokens):
                deficit = float(estimated_tokens) - self.token_tokens
                wait_tok = deficit / self.token_fill_rate
                needed_wait = max(needed_wait, wait_tok)

            if needed_wait <= 0.001:
                # Deduct from per-minute bucket
                self.req_tokens -= 1.0
                self.token_tokens -= float(estimated_tokens)
                return

            self.sleep_fn(needed_wait)

    def record_usage(self, actual_tokens_in: int, actual_tokens_out: int) -> None:
        """Record actual tokens consumed after successful API call."""
        total = actual_tokens_in + actual_tokens_out
        self._check_and_reset_day()
        self.daily_requests_used += 1
        self.daily_tokens_used += total
        self._save_state()

    def update_from_headers(self, headers: dict[str, Any]) -> None:
        """Sync internal counters with authoritative Groq response headers."""
        # Remaining requests / tokens per minute
        rem_req = headers.get("x-ratelimit-remaining-requests")
        rem_tok = headers.get("x-ratelimit-remaining-tokens")
        if rem_req is not None:
            try:
                self.req_tokens = min(self.req_capacity, float(rem_req))
            except (ValueError, TypeError):
                pass
        if rem_tok is not None:
            try:
                self.token_tokens = min(self.token_capacity, float(rem_tok))
            except (ValueError, TypeError):
                pass

    def handle_429(self, retry_after_s: float | None = None) -> None:
        """Handle HTTP 429 Too Many Requests by backoff sleeping."""
        wait = retry_after_s if (retry_after_s and retry_after_s > 0) else 5.0
        self.req_tokens = 0.0
        self.sleep_fn(wait)
        self.last_leak_time = self.time_fn()
