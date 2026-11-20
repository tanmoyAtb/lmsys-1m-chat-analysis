"""K-means topic clustering of the 5,000 sample, following the LMSYS-Chat-1M paper (Sec. 3.2).

Same steps as the paper:
  1. user prompts, initial AND follow-up turns
  2. drop prompts < 32 or > 1536 characters (filter applies to each message, never to a
     whole conversation)
  3. embed each prompt with SentenceTransformers all-mpnet-base-v2
  4. k-means, k = 20
  5. name each cluster from the 100 items closest to its centroid (paper: GPT-4; here OPENAI_MODEL)

Two versions from the SAME kept prompts and embeddings:
  A  prompt-level        one point per kept prompt (the paper's unit)
  B  conversation-level  one point per conversation = mean of its kept prompts' embeddings
                         (conversations with no kept prompt cannot be embedded and are excluded)

Settings the paper does not specify (chosen here): embeddings L2-normalized (cosine geometry;
B re-normalized after averaging), KMeans(n_init=10, random_state=42). Clusters are numbered
1..20 by size, largest first, like the paper's Figure 3.

Outputs (analysis/clustering/):
  prompts_A.csv           conversation_id, msg_index, chars, cluster
  conversations_B.csv     conversation_id, n_prompts, cluster
  clusters_A.csv / _B.csv cluster, size, percent, title, description
  clusters_A.md  / _B.md  title, description and closest examples per cluster (for mapping)
Embeddings are cached in data/embeddings/ (git-ignored; reproducible).

Usage:
  uv run python scripts/cluster.py                 # embed (cached), cluster, name
  uv run python scripts/cluster.py --skip-naming   # no OpenAI calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = PROJECT_ROOT / "data" / "sample" / "sample_english_5000_seed42.parquet"
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
OUT_DIR = PROJECT_ROOT / "analysis" / "clustering"

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
MIN_CHARS, MAX_CHARS = 32, 1536
K = 20
SEED = 42
N_NAMING = 100  # items closest to each centroid sent for naming (paper: 100)
N_EXAMPLES_MD = 15  # examples shown per cluster in the .md files
NAMING_ITEM_CHARS = 400  # each naming item is truncated to this length

NAMING_PROMPT = """Below are {n} user prompts sent to AI chatbots. They all belong to one cluster \
produced by k-means on sentence embeddings, and are the prompts closest to the cluster centroid.

Summarize the central topic of this cluster.
- "title": a short descriptive title, at most 10 words (e.g. "Requests for Python coding assistance and examples").
- "description": one sentence describing what these prompts have in common.
The prompts may contain offensive or sexual text; you are only describing them.

PROMPTS:
{items}"""


# ---------------------------------------------------------------- data
def extract_prompts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        for i, m in enumerate(r.conversation):
            if m["role"] == "user":
                text = m["content"] or ""
                rows.append((r.conversation_id, i, len(text), text))
    p = pd.DataFrame(rows, columns=["conversation_id", "msg_index", "chars", "text"])
    p["keep"] = p["chars"].between(MIN_CHARS, MAX_CHARS)
    return p


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    return model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def cached_embeddings(prompts: pd.DataFrame) -> np.ndarray:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    key = prompts["conversation_id"] + ":" + prompts["msg_index"].astype(str)
    emb_path, key_path = EMB_DIR / "prompts_mpnet.npy", EMB_DIR / "prompts_mpnet_keys.txt"
    if emb_path.exists() and key_path.exists() and key_path.read_text().split("\n") == key.tolist():
        print(f"Using cached embeddings ({emb_path.relative_to(PROJECT_ROOT)})")
        return np.load(emb_path)
    print(f"Embedding {len(prompts):,} prompts with {EMBED_MODEL} ...")
    X = embed(prompts["text"].tolist())
    np.save(emb_path, X)
    key_path.write_text("\n".join(key))
    return X


# ---------------------------------------------------------------- clustering
def kmeans_by_size(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit k-means; renumber clusters 1..K by size (largest first). Returns labels, distances."""
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
    sizes = np.bincount(km.labels_, minlength=K)
    rank = {old: new + 1 for new, old in enumerate(np.argsort(-sizes, kind="stable"))}
    labels = np.array([rank[c] for c in km.labels_])
    dist = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    return labels, dist


def closest(labels: np.ndarray, dist: np.ndarray, texts: list[str], c: int, n: int) -> list[str]:
    idx = np.where(labels == c)[0]
    return [texts[i] for i in idx[np.argsort(dist[idx])][:n]]


# ---------------------------------------------------------------- naming
def name_clusters(examples: dict[int, list[str]]) -> dict[int, dict]:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_BASE_URL"):
        os.environ.pop("OPENAI_BASE_URL", None)
    model = os.getenv("OPENAI_MODEL")
    if not os.getenv("OPENAI_API_KEY") or not model:
        sys.exit("OPENAI_API_KEY and OPENAI_MODEL must be set in .env (or use --skip-naming).")
    client = OpenAI(max_retries=6, timeout=180)
    schema = {
        "name": "cluster_name", "strict": True,
        "schema": {"type": "object", "additionalProperties": False, "required": ["title", "description"],
                   "properties": {"title": {"type": "string"}, "description": {"type": "string"}}},
    }

    def one(c: int) -> tuple[int, dict]:
        items = "\n".join(f"{i + 1}. {t[:NAMING_ITEM_CHARS].replace(chr(10), ' ')}"
                          for i, t in enumerate(examples[c]))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": NAMING_PROMPT.format(n=len(examples[c]), items=items)}],
            response_format={"type": "json_schema", "json_schema": schema},
        )
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            return c, {"title": "(naming refused)", "description": msg.refusal[:200]}
        return c, json.loads(msg.content)

    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(one, sorted(examples)))


# ---------------------------------------------------------------- outputs
def write_summary(version: str, unit: str, labels: np.ndarray, names: dict[int, dict],
                  examples: dict[int, list[str]]) -> pd.DataFrame:
    sizes = pd.Series(labels).value_counts().sort_index()
    summary = pd.DataFrame({
        "cluster": sizes.index,
        "size": sizes.values,
        "percent": (sizes.values / sizes.sum() * 100).round(2),
        "title": [names.get(c, {}).get("title", "") for c in sizes.index],
        "description": [names.get(c, {}).get("description", "") for c in sizes.index],
    })
    summary.to_csv(OUT_DIR / f"clusters_{version}.csv", index=False)

    lines = [f"# Clusters {version}: {unit}", "",
             f"k-means k={K} on {EMBED_MODEL} embeddings; {sizes.sum():,} {unit}s. "
             f"Examples are the {N_EXAMPLES_MD} closest to each centroid (truncated).", ""]
    for r in summary.itertuples():
        lines += [f"## Cluster {r.cluster}: {r.title or '(unnamed)'}  ({r.percent}%, n={r.size})", ""]
        if r.description:
            lines += [f"_{r.description}_", ""]
        for t in examples[r.cluster][:N_EXAMPLES_MD]:
            t = " ".join(t.split())
            lines.append(f"- {t[:300]}{'…' if len(t) > 300 else ''}")
        lines.append("")
    (OUT_DIR / f"clusters_{version}.md").write_text("\n".join(lines))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-naming", action="store_true", help="don't call OpenAI to name clusters")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SAMPLE, columns=["conversation_id", "conversation"])
    prompts_all = extract_prompts(df)
    prompts = prompts_all[prompts_all["keep"]].reset_index(drop=True)
    print(f"User prompts: {len(prompts_all):,}; kept {len(prompts):,} "
          f"({MIN_CHARS}-{MAX_CHARS} chars per message)")

    X = cached_embeddings(prompts)

    # ---- A: prompt-level
    labels_a, dist_a = kmeans_by_size(X)
    prompts["cluster"] = labels_a
    prompts[["conversation_id", "msg_index", "chars", "cluster"]].to_csv(OUT_DIR / "prompts_A.csv", index=False)
    texts_a = prompts["text"].tolist()
    ex_a = {c: closest(labels_a, dist_a, texts_a, c, N_NAMING) for c in range(1, K + 1)}

    # ---- B: conversation-level (mean of kept prompt embeddings, re-normalized)
    codes, conv_ids = pd.factorize(prompts["conversation_id"])
    Xb = np.zeros((len(conv_ids), X.shape[1]), dtype=np.float64)
    np.add.at(Xb, codes, X)
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    labels_b, dist_b = kmeans_by_size(Xb.astype(np.float32))
    n_prompts = np.bincount(codes)
    conv = pd.DataFrame({"conversation_id": conv_ids, "n_prompts": n_prompts, "cluster": labels_b})
    conv.to_csv(OUT_DIR / "conversations_B.csv", index=False)
    joined = prompts.groupby("conversation_id")["text"].apply(lambda s: " ⏎ ".join(s))
    texts_b = joined.loc[conv_ids].tolist()
    ex_b = {c: closest(labels_b, dist_b, texts_b, c, N_NAMING) for c in range(1, K + 1)}
    print(f"A: {len(prompts):,} prompts | B: {len(conv):,} conversations "
          f"({df['conversation_id'].nunique() - len(conv):,} have no kept prompt)")

    # ---- naming
    names_a = names_b = {}
    if not args.skip_naming:
        print("Naming clusters ...")
        names_a, names_b = name_clusters(ex_a), name_clusters(ex_b)

    for version, unit, labels, names, ex in [("A", "prompt", labels_a, names_a, ex_a),
                                             ("B", "conversation", labels_b, names_b, ex_b)]:
        s = write_summary(version, unit, labels, names, ex)
        print(f"\nClusters {version} ({unit}-level):")
        print(s[["cluster", "size", "percent", "title"]].to_string(index=False))
    print(f"\nWrote {OUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
