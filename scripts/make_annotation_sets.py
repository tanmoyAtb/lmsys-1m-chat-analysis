"""Create the two human-annotation sets from the 5,000-conversation sample.

  random60   60 conversations drawn uniformly at random from the sample.
             Used to compare category percentages (human vs LLM vs k-means vs paper),
             so it must be an unbiased random draw.
  flagged40  40 conversations selected ONLY via the OpenAI moderation flags
             (at least one message flagged), excluding the random60 ones.
             Used to study traits. To get variety instead of 40 near-identical
             sexual-content conversations, selection is stratified by flagged
             category: categories are visited rarest-first in round-robin, and
             each visit draws one random not-yet-chosen conversation flagged
             with that category.

Both sets keep all columns of the source sample. Outputs go next to the sample
(data/sample/) so the explorer app can open them.

Usage:
  uv run python scripts/make_annotation_sets.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = PROJECT_ROOT / "data" / "sample" / "sample_english_5000_seed42.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "sample"
SEED = 42
N_RANDOM = 60
N_FLAGGED = 40


def flagged_categories(mods) -> list[str]:
    # Only messages with flagged=True count. (In the raw data ~8% of conversations have a
    # category set to True on a message whose overall `flagged` is False; those are ignored.)
    return sorted({c for m in mods if m["flagged"] for c, v in m["categories"].items() if v})


def main() -> None:
    df = pd.read_parquet(SAMPLE)
    rng = np.random.default_rng(SEED)

    # --- random60: uniform random draw from the full sample
    random_ids = rng.choice(np.sort(df["conversation_id"].to_numpy()), size=N_RANDOM, replace=False)

    # --- flagged40: moderation flags only, disjoint from random60
    cats = df.set_index("conversation_id")["openai_moderation"].map(flagged_categories)
    pool = cats[(cats.map(len) > 0) & ~cats.index.isin(random_ids)]
    by_cat: dict[str, list[str]] = {}
    for cid, cs in pool.items():
        for c in cs:
            by_cat.setdefault(c, []).append(cid)
    for c in by_cat:  # random order within each category
        by_cat[c] = list(rng.permutation(sorted(by_cat[c])))
    order = sorted(by_cat, key=lambda c: (len(by_cat[c]), c))  # rarest first

    chosen: list[str] = []
    while len(chosen) < N_FLAGGED and any(by_cat.values()):
        for c in order:
            while by_cat[c] and by_cat[c][0] in chosen:
                by_cat[c].pop(0)
            if by_cat[c] and len(chosen) < N_FLAGGED:
                chosen.append(by_cat[c].pop(0))
    flagged_ids = np.array(chosen)

    assert len(set(random_ids) & set(flagged_ids)) == 0

    # --- write
    sets = {"random60": random_ids, "flagged40": flagged_ids}
    for name, ids in sets.items():
        sub = df.set_index("conversation_id").loc[list(ids)].reset_index()
        stem = f"annot_{name}_seed{SEED}"
        sub.to_parquet(OUT_DIR / f"{stem}.parquet", compression="zstd", index=False)
        (OUT_DIR / f"{stem}_ids.txt").write_text("\n".join(ids) + "\n")

    sub40 = cats.loc[list(flagged_ids)]
    meta = {
        "source": SAMPLE.name,
        "seed": SEED,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random60": "uniform random sample without replacement from the 5,000 sample (sorted ids, "
                    "numpy default_rng(seed).choice)",
        "flagged40": "conversations with >=1 OpenAI-moderation-flagged message, excluding random60; "
                     "stratified round-robin over flagged categories (rarest first), random within category",
        "flagged_pool_size": int(len(pool)),
        "random60_flagged_count": int((cats.loc[list(random_ids)].map(len) > 0).sum()),
        "flagged40_category_coverage": pd.Series([c for cs in sub40 for c in cs]).value_counts().to_dict(),
    }
    (OUT_DIR / f"annot_sets_seed{SEED}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
