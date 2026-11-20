"""Propose which of the paper's 20 topics each of our k-means clusters corresponds to.

For every cluster in clustering A and B, the LLM (OPENAI_MODEL) sees the paper's 20 topic names
and the cluster's closest examples (recomputed from the cached embeddings; start and end of long prompts) and answers
with one paper topic number, or "none" if the cluster is not the SAME topic.

Writes analysis/clustering/paper_topics.yaml for human review:

  A:
    8:  # ours: Python Programming Help and Code Generation (5.8%)
      paper_topic: 6  # Requests for Python coding assistance and examples | LLM: high - <reason>

Edit `paper_topic` (a paper cluster number, or none). The file is not overwritten unless
--overwrite is given, so a reviewed file is safe.

Usage:
  uv run python scripts/match_paper_topics.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CL = ROOT / "analysis" / "clustering"
OUT = CL / "paper_topics.yaml"

PROMPT = """You are aligning topic clusters from two analyses of the same kind of chatbot data.

The REFERENCE topics are the 20 clusters of user prompts from a published paper:
{paper}

Below is ONE cluster from our analysis: its title and the user prompts closest to its centroid.
Decide which reference topic it is the SAME topic as.
- Answer with a reference number only if most of the examples would clearly belong in that reference
  topic (same kind of request, or the same scripted template). Merely related is NOT enough.
- Answer "none" if no reference topic fits that well.
- confidence: high / medium / low. reason: one short sentence.

OUR CLUSTER: {title}
EXAMPLES:
{examples}"""


N_EXAMPLES = 20


def head_tail(text: str, head: int = 120, tail: int = 300) -> str:
    """Scripted templates put the real request at the END, so keep both ends."""
    t = " ".join(text.split())
    return t if len(t) <= head + tail else f"{t[:head]} … {t[-tail:]}"


def load_clusters() -> dict[str, dict[int, dict]]:
    """Titles from clusters_{A,B}.csv; examples = items closest to each centroid, recomputed from
    the cached embeddings exactly as scripts/cluster.py builds them."""
    sample = pd.read_parquet(ROOT / "data/sample/sample_english_5000_seed42.parquet",
                             columns=["conversation_id", "conversation"]).set_index("conversation_id")
    pa = pd.read_csv(CL / "prompts_A.csv")
    pa["text"] = [sample.loc[c, "conversation"][i]["content"] for c, i in zip(pa["conversation_id"], pa["msg_index"])]
    X = np.load(ROOT / "data/embeddings/prompts_mpnet.npy")
    assert len(X) == len(pa), "embeddings do not match prompts_A.csv; re-run scripts/cluster.py"

    codes, conv_ids = pd.factorize(pa["conversation_id"])
    Xb = np.zeros((len(conv_ids), X.shape[1]))
    np.add.at(Xb, codes, X)
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    cb = pd.read_csv(CL / "conversations_B.csv").set_index("conversation_id").loc[conv_ids]
    joined = pa.groupby("conversation_id")["text"].apply(lambda s: " ⏎ ".join(s)).loc[conv_ids]

    out = {}
    for v, emb, labels, texts in [("A", X, pa["cluster"].to_numpy(), pa["text"].tolist()),
                                  ("B", Xb, cb["cluster"].to_numpy(), joined.tolist())]:
        summary = pd.read_csv(CL / f"clusters_{v}.csv").set_index("cluster")
        out[v] = {}
        for c in summary.index:
            idx = np.where(labels == c)[0]
            centroid = emb[idx].mean(axis=0)
            order = idx[np.argsort(np.linalg.norm(emb[idx] - centroid, axis=1))][:N_EXAMPLES]
            out[v][int(c)] = {"title": summary.loc[c, "title"], "percent": float(summary.loc[c, "percent"]),
                              "examples": [head_tail(texts[i]) for i in order]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.overwrite:
        sys.exit(f"{OUT.relative_to(ROOT)} exists (maybe already reviewed). Use --overwrite to replace it.")

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_BASE_URL"):
        os.environ.pop("OPENAI_BASE_URL", None)
    model = os.getenv("OPENAI_MODEL")
    if not os.getenv("OPENAI_API_KEY") or not model:
        sys.exit("OPENAI_API_KEY and OPENAI_MODEL must be set in .env.")
    client = OpenAI(max_retries=6, timeout=180)

    paper = pd.read_csv(ROOT / "reference" / "paper_fig3_topics.csv").set_index("cluster")
    paper_txt = "\n".join(f"{c}. {r.label}" for c, r in paper.iterrows())
    options = [str(c) for c in paper.index] + ["none"]
    schema = {"name": "topic_match", "strict": True, "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["paper_topic", "confidence", "reason"],
        "properties": {"paper_topic": {"type": "string", "enum": options},
                       "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                       "reason": {"type": "string"}}}}

    def ask(item: tuple[str, int, dict]) -> tuple[str, int, dict]:
        version, c, info = item
        ex = "\n".join(f"- {e}" for e in info["examples"])
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT.format(paper=paper_txt, title=info["title"], examples=ex)}],
            response_format={"type": "json_schema", "json_schema": schema},
        )
        return version, c, json.loads(resp.choices[0].message.content)

    clusters = load_clusters()
    items = [(v, c, info) for v, cs in clusters.items() for c, info in cs.items()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = {(v, c): a for v, c, a in pool.map(ask, items)}

    lines = [
        "# Which of the paper's 20 topics each of our clusters is the SAME topic as.",
        "# Proposed by the LLM (scripts/match_paper_topics.py); REVIEW and edit `paper_topic`:",
        "# a paper cluster number (1-20), or none. Several clusters may share one paper topic.",
        "#", "# Paper topics:", *[f"#   {c:>2}. {r.label}" for c, r in paper.iterrows()], "",
    ]
    for v in ("A", "B"):
        lines.append(f"{v}:  # {'prompt' if v == 'A' else 'conversation'}-level clusters (see clusters_{v}.md)")
        for c, info in clusters[v].items():
            a = answers[(v, c)]
            pt = a["paper_topic"]
            label = paper.loc[int(pt), "label"] if pt != "none" else "no matching paper topic"
            reason = a["reason"].replace("\n", " ").strip()
            lines.append(f"  {c}:  # ours: {info['title']} ({info['percent']}%)")
            lines.append(f"    paper_topic: {pt}  # {label} | LLM: {a['confidence']} - {reason}")
        lines.append("")
    OUT.write_text("\n".join(lines))

    for v in ("A", "B"):
        print(f"\n{v}:")
        for c, info in clusters[v].items():
            a = answers[(v, c)]
            print(f"  {c:>2} {info['title'][:52]:52} -> {a['paper_topic']:>4} ({a['confidence']})")
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
