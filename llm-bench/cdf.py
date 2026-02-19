from __future__ import annotations

from pathlib import Path
from typing import Sequence


def write_cdf_csv(values: Sequence[float] | Sequence[int], out_csv: Path, *, kind: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if kind == "percentile":
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("percentile,value_s\n")
            if not values:
                for p in range(0, 101):
                    f.write(f"{p},\n")
                return
            s = sorted(float(v) for v in values)
            for p in range(0, 101):
                idx = int(p / 100 * (len(s) - 1))
                f.write(f"{p},{s[idx]:.9f}\n")
        return

    if kind == "empirical":
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("accumulated_percentage,batch_size\n")
            if not values:
                return
            s = sorted(int(v) for v in values)
            n = len(s)
            for i, v in enumerate(s):
                is_last = i == n - 1
                if not is_last and s[i + 1] == v:
                    continue
                pct = (i + 1) / n * 100.0
                f.write(f"{pct:.6f},{v}\n")
        return

    raise ValueError(f"Unknown CDF kind: {kind!r}")

