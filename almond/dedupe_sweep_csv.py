#!/usr/bin/env python3
"""Collapse duplicate samples in p32's summary file, keeping the better row.

p32 --redo appends a second row for a sample that is already present, which
is how a frozen-only sample gets a fresh reference added after the fact. Two
rows for one sample would double-count it in every median, so this collapses
them on (mesh, ell, level, case, seed), preferring the row that HAS a fresh
solve, and among those the later one.

    python dedupe_sweep_csv.py data/sweep/p32_almond_sweep_summary.csv

Writes a .bak beside the file and reports what it merged. Idempotent.
"""
import csv, os, shutil, sys

path = sys.argv[1]
with open(path) as fh:
    rdr = csv.DictReader(fh)
    hdr, rows = rdr.fieldnames, list(rdr)

best = {}
for i, r in enumerate(rows):
    k = (r["mesh"], r["ell"], r["level"], r["case"], r["seed"])
    # rank: has a fresh solve first, then later row wins
    rank = (int(r["iters_fresh"]) > 0, i)
    if k not in best or rank > best[k][0]:
        best[k] = (rank, r)

keep = [r for _, r in sorted(best.values(), key=lambda t: t[0][1])]
drop = len(rows) - len(keep)
if drop == 0:
    print(f"  no duplicates in {os.path.basename(path)} ({len(rows)} rows)")
    sys.exit(0)

shutil.copy2(path, path + ".bak")
with open(path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=hdr)
    w.writeheader()
    w.writerows(keep)
print(f"  {os.path.basename(path)}: {len(rows)} -> {len(keep)} rows "
      f"({drop} superseded); .bak written")
