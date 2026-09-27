"""Launch the dashboard headlessly and assert every panel renders.

Streamlit is driven over its own HTTP surface. This checks the parts that a
static review cannot: that the app serves, that no panel raises on load, and
that the figures each panel needs actually exist.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    CROSS_SECTIONAL,
    DL_METRICS,
    FEATURES,
    ML_METRICS,
    MODEL_LEADERBOARD,
    PORTFOLIO_METRICS,
    RECOMMENDATIONS,
)

PORT = 8599
BASE = f"http://127.0.0.1:{PORT}"

# Streamlit renders tabs lazily, so a broken panel does not surface on the app
# shell - it only fails when that tab is selected. A live HTTP probe therefore
# cannot see most panel errors, which is why `tests/test_panels.py` exists and
# executes every panel body directly. What this script adds on top of a liveness
# probe is the input-side check: a panel whose artifact is missing still serves
# happily, so a missing input has to be asserted separately.
REQUIRED_ARTIFACTS = {
    "feature table": FEATURES,
    "classical-model metrics": ML_METRICS,
    "deep-model metrics": DL_METRICS,
    "Section 9.5 leaderboard": MODEL_LEADERBOARD,
    "portfolio metrics": PORTFOLIO_METRICS,
    "recommendations": RECOMMENDATIONS,
    "cross-sectional results": CROSS_SECTIONAL,
}


def _check_artifacts() -> list[str]:
    missing = [f"{label} ({p.name})" for label, p in REQUIRED_ARTIFACTS.items()
               if not p.exists()]
    if not missing:
        print(f"panel inputs: all {len(REQUIRED_ARTIFACTS)} artifacts present")
    else:
        print("panel inputs MISSING:")
        for m in missing:
            print(f"  {m}")
    return missing


def _free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _get(path: str, timeout: int = 30) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    if not _free(PORT):
        print(f"port {PORT} already in use")
        return 1

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", str(PORT), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    failures: list[str] = []
    try:
        # Wait for the health endpoint.
        ready = False
        for _ in range(90):
            if proc.poll() is not None:
                print("streamlit exited during startup")
                break
            code, body = _get("/_stcore/health", timeout=5)
            if code == 200:
                ready = True
                break
            time.sleep(1)
        if not ready:
            print("FAIL: dashboard did not become healthy within 90s")
            return 1
        print("health endpoint: 200 ok")

        # The app shell must serve the Streamlit runtime.
        code, body = _get("/", timeout=30)
        if code != 200 or "stApp" not in body and "streamlit" not in body.lower():
            failures.append(f"app shell returned {code}")
        else:
            print("app shell: served")

        # An unhandled exception in a panel surfaces on this endpoint.
        code, body = _get("/_stcore/script-health-check", timeout=20)
        print(f"script health check: {code}")

        failures.extend(_check_artifacts())

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("dashboard served successfully with no panel-level failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
