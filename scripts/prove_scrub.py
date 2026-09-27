"""Prove the clean-env report scrubber cannot leak a path.

Three earlier bugs came out of writing this by hand rather than testing it:
an over-escaped drive-letter pattern that matched nothing, a body that stopped
at the first space (leaving the rest of the path in a spaced username), and a
scrubber that was only ever exercised on POSIX-style input. This exercises all
three shapes, and fails loudly if any of them would ship.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "clean_env_check.py"
spec = importlib.util.spec_from_file_location("cec", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Path-shaped fixtures, assembled at runtime rather than written literally.
# A committed proof script must not contain a real home directory, and a
# placeholder is not path-shaped enough to exercise the greedy match body.
_SEP = "\\"
_HOME = _SEP.join(["C:", "Users", "some user", "AppData", "Local", "Temp"])
_PX = _SEP.join(["C:", "Users", "some user", "Desktop", "proj", "src", "x.py"])

CASES = [
    f'Actual location:    "{_HOME}{_SEP}venv"',
    f'  File "{_PX}", line 3',
    f"{_HOME}{_SEP}venv{_SEP}Scripts{_SEP}python.exe -m pytest",
    "site-packages at /home/runner/work/repo/venv/lib",
    "cloned to /Users/someone/Desktop/proj",
    "/root/.cache/pip/wheels",
    "plain line with no path at all",
    "FINNHUB_API_KEY is not set",
]

# Anything matching these in the output would be a real leak.
FORBIDDEN = ("some user", "AppData", "Local", "Desktop",
             "Users\\", "Users/", "/home/", "/Users/", "/root/")

print("=== scrubber output ===")
leaks = []
for case in CASES:
    out = m._clean(case, limit=5)
    rendered = out[0] if out else "(dropped)"
    print(f"  in : {case[:70]}")
    print(f"  out: {rendered[:70]}")
    for bad in FORBIDDEN:
        if bad in rendered:
            leaks.append((case, bad, rendered))

print()
if leaks:
    print("LEAK - the following survived redaction:")
    for case, bad, rendered in leaks:
        print(f"  {bad!r} in: {case[:60]}")
        print(f"           -> {rendered[:70]}")
    raise SystemExit(1)

print("PASS - no username, home directory or absolute path survives")
print("       across Windows, spaced-username and POSIX inputs.")
