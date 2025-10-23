# scripts/split_ratings_into_batches.py
from __future__ import annotations

import csv
import math
from pathlib import Path

# ========= EDIT ME (defaults) =========
# Paths are relative to repo root; change as you like.
INPUT_CSV   = Path(__file__).resolve().parents[2] / "data" / "raw" / "ml-20m" / "ratings.csv"
OUTDIR      = Path(__file__).resolve().parents[2] / "data" / "incoming"
N_BATCHES   = 250
PREFIX      = "ratings_batch_"
# =====================================

def _count_data_rows(path: Path) -> int:
    """Count lines minus the header (assumes exactly one header line)."""
    with path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline()
        if not header:
            return 0
        return sum(1 for _ in f)

def _pad(n: int) -> int:
    """Width for zero-padding (001..)."""
    return max(3, len(str(n)))

def split_csv(input_csv: Path, outdir: Path, n_batches: int, prefix: str = "ratings_batch_") -> None:
    if n_batches < 1:
        raise ValueError("N_BATCHES must be >= 1")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    outdir.mkdir(parents=True, exist_ok=True)

    total_rows = _count_data_rows(input_csv)
    if total_rows == 0:
        print("Input has no data rows (maybe only header) — nothing to do.")
        return

    pad = _pad(n_batches)
    rows_per_batch = math.ceil(total_rows / n_batches)

    print(f"Splitting {input_csv}")
    print(f"→ Output dir: {outdir}")
    print(f"→ Batches: {n_batches}  (~{rows_per_batch:,} rows per batch)")
    print()

    current_batch = 1
    current_written = 0

    def open_writer(batch_idx: int):
        fname = f"{prefix}{batch_idx:0{pad}d}.csv"
        p = outdir / fname
        f = p.open("w", encoding="utf-8", newline="")
        w = csv.writer(f)
        w.writerow(header)  # write header in each file
        return f, w, p

    with input_csv.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin)
        header = next(reader)  # assume header exists

        f_out, writer, out_path = open_writer(current_batch)
        print(f"→ writing {out_path.name}")

        try:
            for row in reader:
                writer.writerow(row)
                current_written += 1

                if current_written >= rows_per_batch and current_batch < n_batches:
                    f_out.close()
                    print(f"  batch {current_batch:0{pad}d} done ({current_written:,} rows)")
                    current_batch += 1
                    current_written = 0
                    f_out, writer, out_path = open_writer(current_batch)
                    print(f"→ writing {out_path.name}")
        finally:
            f_out.close()

    created = sorted(p.name for p in outdir.glob(f"{prefix}*.csv"))
    print("\n✅ Done.")
    print(f"Created {len(created)} files. First few: {', '.join(created[:5])}{' ...' if len(created) > 5 else ''}")

if __name__ == "__main__":
    # Primary mode: use the config block above
    split_csv(INPUT_CSV, OUTDIR, N_BATCHES, PREFIX)
