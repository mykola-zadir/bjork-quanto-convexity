"""Run every tests/test_*.py and report one grand total (RED on any failure)."""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT_RE = re.compile(r"RESULT:\s+(\d+)\s+passed,\s+(\d+)\s+failed")

if __name__ == "__main__":
    tot_p = tot_f = err = 0
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        proc = subprocess.run([sys.executable, str(f)], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=600)
        m = list(RESULT_RE.finditer(proc.stdout + proc.stderr))
        if not m:
            print(f"  {f.name:<34} CRASHED"); err += 1; continue
        p, fl = int(m[-1].group(1)), int(m[-1].group(2))
        tot_p += p; tot_f += fl
        print(f"  {f.name:<34}{p:>4} passed{fl:>4} failed")
    ok = tot_f == 0 and err == 0 and tot_p > 0
    print(f"\n  GRAND TOTAL: {tot_p} passed, {tot_f} failed, {err} errored")
    print(f"  SUITE: {'GREEN' if ok else 'RED'}")
    sys.exit(0 if ok else 1)
