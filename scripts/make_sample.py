"""Draw a reproducible random sample of conversations from LMSYS-Chat-1M.

By default: 5,000 conversations, uniformly at random (without replacement)
from all conversations whose `language` field is "English", seed 42.
All original columns are kept (full conversation, model, turn, moderation, ...).

Outputs (in --out-dir, default data/sample/):
  sample_<lang>_<n>_seed<seed>.parquet   full rows of the sampled conversations
  sample_<lang>_<n>_seed<seed>_ids.txt   sampled conversation_ids (one per line)
  sample_<lang>_<n>_seed<seed>_meta.json how the sample was made

Usage:
  uv run python scripts/make_sample.py
  uv run python scripts/make_sample.py --n 5000 --seed 42 --language English
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "lmsys-chat-1m"
DEFAULT_OUT = PROJECT_ROOT / "data" / "sample"


def dataset_revision(raw_dir: Path) -> str | None:
    """Commit hash of the downloaded HF dataset, read from the hf download cache."""
    meta = sorted((raw_dir / ".cache" / "huggingface" / "download" / "data").glob("*.metadata"))
    revs = {m.read_text().splitlines()[0].strip() for m in meta if m.read_text().strip()}
    return revs.pop() if len(revs) == 1 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000, help="number of conversations to sample")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--language", default="English", help='value of the `language` column to keep ("" = all)')
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    files = sorted((args.raw_dir / "data").glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {args.raw_dir / 'data'}. Run scripts/download_data.sh first.")

    # 1) Build the candidate pool (cheap: only id + language columns).
    pool_ids = []
    total = 0
    for f in files:
        t = pq.read_table(f, columns=["conversation_id", "language"])
        total += t.num_rows
        if args.language:
            t = t.filter(pc.equal(t["language"], args.language))
        pool_ids.append(t["conversation_id"])
    pool = pa.chunked_array(pool_ids).to_numpy(zero_copy_only=False)
    if len(np.unique(pool)) != len(pool):
        raise SystemExit("conversation_id is not unique in the pool; cannot sample by id safely.")
    if args.n > len(pool):
        raise SystemExit(f"Requested n={args.n} but pool has only {len(pool):,} conversations.")
    print(f"Pool: {len(pool):,} of {total:,} conversations (language={args.language or 'ALL'})")

    # 2) Uniform random sample without replacement. Sorting the pool first makes the
    #    result independent of file/row order.
    rng = np.random.default_rng(args.seed)
    pool.sort()
    chosen = rng.choice(pool, size=args.n, replace=False)
    chosen_set = pa.array(chosen)

    # 3) Fetch the full rows for the chosen ids.
    parts = []
    for f in files:
        t = pq.read_table(f)
        parts.append(t.filter(pc.is_in(t["conversation_id"], value_set=chosen_set)))
    sample = pa.concat_tables(parts)
    assert sample.num_rows == args.n, f"expected {args.n} rows, got {sample.num_rows}"

    # Keep rows in the order they were drawn (random order, reproducible).
    order = {cid: i for i, cid in enumerate(chosen)}
    idx = np.argsort([order[c] for c in sample["conversation_id"].to_pylist()])
    sample = sample.take(pa.array(idx))

    # 4) Write outputs.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{(args.language or 'all').lower()}_{args.n}_seed{args.seed}"
    pq.write_table(sample, args.out_dir / f"{stem}.parquet", compression="zstd")
    (args.out_dir / f"{stem}_ids.txt").write_text("\n".join(chosen) + "\n")
    meta = {
        "source": "lmsys/lmsys-chat-1m",
        "source_revision": dataset_revision(args.raw_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "uniform random sample without replacement over sorted conversation_ids "
                  "after language filter; numpy default_rng(seed).choice",
        "language_filter": args.language or None,
        "seed": args.seed,
        "n": args.n,
        "pool_size": int(len(pool)),
        "dataset_size": int(total),
        "columns": sample.column_names,
    }
    (args.out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    size_mb = (args.out_dir / f"{stem}.parquet").stat().st_size / 1e6
    print(f"Wrote {args.n:,} conversations -> {args.out_dir / (stem + '.parquet')} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
