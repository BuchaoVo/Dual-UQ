from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

USER_AGENT = "dual-uq-inverse-folding/0.1 (research pipeline)"


def request_json(
    url: str,
    *,
    timeout: int = 30,
    retries: int = 2,
    backoff_seconds: float = 1.5,
) -> Any:
    last_error: Exception | None = None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(backoff_seconds * (attempt + 1))

    raise RuntimeError(
        f"Failed to retrieve JSON after {retries} attempts "
        f"(timeout={timeout}s): {url}"
    ) from last_error


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int = 30,
    retries: int = 2,
    backoff_seconds: float = 1.5,
) -> Any:
    last_error: Exception | None = None
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(backoff_seconds * (attempt + 1))

    raise RuntimeError(
        f"Failed to POST JSON after {retries} attempts "
        f"(timeout={timeout}s): {url}"
    ) from last_error


def download_file(
    url: str,
    output_path: str | Path,
    *,
    timeout: int = 120,
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and output.stat().st_size > 0:
        return output

    temporary = output.with_suffix(output.suffix + ".part")
    headers = {"User-Agent": USER_AGENT}
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            with requests.get(
                url,
                timeout=timeout,
                headers=headers,
                stream=True,
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)

            if temporary.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty file: {url}")

            temporary.replace(output)
            return output
        except (requests.RequestException, OSError, RuntimeError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(backoff_seconds * (attempt + 1))

    raise RuntimeError(f"Failed to download after {retries} attempts: {url}") from last_error
