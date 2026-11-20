"""Annotation page: human labels for the 100 annotation conversations.

- The two sets (random60 + flagged40) are merged into one fixed shuffled order, so the
  annotator can't tell which set a conversation comes from.
- Moderation flags/scores are hidden to keep the human labels independent.
- Labels follow annotation/labels.yaml and are saved to annotation/human_labels.csv
  (one row per conversation; re-saving overwrites that row).
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from ui.common import (
    CHAT_MAX_WIDTH_PX,
    PROJECT_ROOT,
    SAMPLE_DIR,
    clamp,
    conversation_header,
    inject_css,
    load_sample,
    pill,
    properties_html,
    render_conversation,
)

PANEL_WIDTH_PX = 330
ORDER_SEED = 42
SETS = {
    "random60": SAMPLE_DIR / "annot_random60_seed42.parquet",
    "flagged40": SAMPLE_DIR / "annot_flagged40_seed42.parquet",
}
LABELS_YAML = PROJECT_ROOT / "annotation" / "labels.yaml"
OUT_CSV = PROJECT_ROOT / "annotation" / "human_labels.csv"
ATTR_TITLES = {
    "anthropomorphism": "Anthropomorphism",
    "sycophancy": "Sycophancy",
    "fear_inducing": "Fear-inducing",
    "harmful": "Harmful",
}

inject_css(PANEL_WIDTH_PX)

# ---------------------------------------------------------------- data
missing = [str(p) for p in SETS.values() if not p.exists()]
if missing:
    st.error(f"Missing annotation sets: {missing}. Run `uv run python scripts/make_annotation_sets.py`.")
    st.stop()

spec = yaml.safe_load(LABELS_YAML.read_text())
LABELS_VERSION = str(spec.get("version", ""))
CATEGORIES = [o["name"] for o in spec["category"]["options"]]
ATTRS = list(spec["attributes"])
PEC_OPTIONS = list(spec["prevent_unsafe_content"]["options"])
COLUMNS = ["conversation_id", "set", "category", *ATTRS, "prevent_unsafe_content",
           "notes", "labels_version", "annotator", "annotated_at"]


@st.cache_data
def load_items() -> pd.DataFrame:
    parts = [load_sample(str(p)).assign(set=name) for name, p in SETS.items()]
    df = pd.concat(parts, ignore_index=True).sort_values("conversation_id")
    order = np.random.default_rng(ORDER_SEED).permutation(len(df))
    return df.iloc[order].reset_index(drop=True)


def load_labels() -> pd.DataFrame:
    if OUT_CSV.exists():
        return pd.read_csv(OUT_CSV, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def save_label(record: dict) -> None:
    labels = load_labels()
    labels = labels[labels["conversation_id"] != record["conversation_id"]]
    labels = pd.concat([labels, pd.DataFrame([record])], ignore_index=True)
    order = {cid: i for i, cid in enumerate(items["conversation_id"])}
    labels = labels.sort_values("conversation_id", key=lambda s: s.map(order))[COLUMNS]
    tmp = OUT_CSV.with_suffix(".csv.tmp")
    labels.to_csv(tmp, index=False)
    tmp.replace(OUT_CSV)


items = load_items()
n = len(items)
labels = load_labels()
saved = labels.set_index("conversation_id").to_dict("index")

# ---------------------------------------------------------------- navigation
if "ann_no" not in st.session_state:
    unlabeled = [i for i, c in enumerate(items["conversation_id"], 1) if c not in saved]
    st.session_state.ann_no = unlabeled[0] if unlabeled else 1
st.session_state.ann_no = clamp(st.session_state.ann_no, n)


def go(delta: int) -> None:
    st.session_state.ann_no = clamp(st.session_state.ann_no + delta, n)


def go_next_unlabeled() -> None:
    done = set(load_labels()["conversation_id"])
    cur = st.session_state.ann_no
    for step in range(1, n + 1):
        i = (cur - 1 + step) % n + 1
        if items["conversation_id"].iloc[i - 1] not in done:
            st.session_state.ann_no = i
            return


row = items.iloc[st.session_state.ann_no - 1]
cid = row["conversation_id"]
prev = saved.get(cid, {})
k = lambda name: f"{name}::{cid}"  # noqa: E731  per-conversation widget keys


def save_and_next(cid: str, set_name: str) -> None:
    ss = st.session_state
    record = {
        "conversation_id": cid,
        "set": set_name,
        "category": ss[f"category::{cid}"],
        **{a: "Yes" if ss.get(f"{a}::{cid}") else "No" for a in ATTRS},
        "prevent_unsafe_content": ss[f"pec::{cid}"],
        "notes": (ss.get(f"notes::{cid}") or "").strip(),
        "labels_version": LABELS_VERSION,
        "annotator": "human",
        "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_label(record)
    ss.ann_no = clamp(ss.ann_no + 1, n)


# ---------------------------------------------------------------- sidebar (left)
with st.sidebar:
    n_done = sum(c in saved for c in items["conversation_id"])
    st.markdown(f'<div class="section">Progress · {n_done} / {n} labeled</div>', unsafe_allow_html=True)
    st.progress(n_done / n)

    status = pill("labeled", "green") if cid in saved else pill("not labeled", "gray")
    st.markdown(
        f'<div class="section">Conversation {st.session_state.ann_no} / {n} &nbsp;{status}</div>'
        + properties_html(row, show_moderation=False),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    render_md = st.toggle("Render markdown", value=True, help="Off shows the raw text exactly as stored.")

    with st.expander("Guidelines"):
        st.markdown(f"**Unit:** {spec['unit']}")
        st.markdown("\n".join(f"- {r}" for r in spec["rules"]))
        st.markdown(f"**Category precedence:** {spec['category']['precedence']}")
        st.caption(f"labels.yaml version {LABELS_VERSION}")

# ---------------------------------------------------------------- main: conversation + annotation panel
col_chat, col_panel = st.columns([CHAT_MAX_WIDTH_PX, PANEL_WIDTH_PX], gap="medium")

with col_chat:
    conversation_header(f"Conversation {st.session_state.ann_no} / {n}", row)
    render_conversation(row, render_md, show_moderation=False)

with col_panel:
    # navigation
    st.markdown("<div style='height:2.1rem'></div>", unsafe_allow_html=True)  # align with chat box
    c_in, c_prev, c_next = st.columns([1.3, 1, 1], vertical_alignment="bottom", gap="small")
    c_in.number_input("Conversation", min_value=1, max_value=n, step=1, key="ann_no",
                      label_visibility="collapsed")
    c_prev.button("◀", on_click=go, args=(-1,), width="stretch", help="Previous")
    c_next.button("▶", on_click=go, args=(1,), width="stretch", help="Next")
    st.button("Next unlabeled ⏭", on_click=go_next_unlabeled, width="stretch", disabled=n_done == n)

    with st.container(border=True):
        # category
        cat_help = "\n".join(
            f"- **{o['name']}** — {o['definition']}" for o in spec["category"]["options"]
        ) + f"\n\n*Precedence:* {spec['category']['precedence']}"
        category = st.radio(
            "**Category**",
            CATEGORIES,
            index=CATEGORIES.index(prev["category"]) if prev.get("category") in CATEGORIES else None,
            key=k("category"),
            help=cat_help,
        )

        # attributes
        st.markdown("**Attributes** <span style='color:#888;font-size:0.85em'>(model replies · checked = Yes)</span>",
                    unsafe_allow_html=True)
        attr_values = {}
        for a in ATTRS:
            d = spec["attributes"][a]
            attr_values[a] = st.checkbox(
                ATTR_TITLES.get(a, a),
                value=prev.get(a) == "Yes",
                key=k(a),
                help=f"{d['question']}\n\n**Yes:** {d['yes_if']}\n\n**No:** {d['no_if']}",
            )

        # unsafe content prevention
        pec_spec = spec["prevent_unsafe_content"]
        pec = st.radio(
            "**Prevent unsafe content**",
            PEC_OPTIONS,
            index=PEC_OPTIONS.index(prev["prevent_unsafe_content"])
            if prev.get("prevent_unsafe_content") in PEC_OPTIONS else None,
            horizontal=True,
            key=k("pec"),
            help=f"{pec_spec['question']}\n\n*Unsafe:* {pec_spec['unsafe_means']}\n\n"
            + "\n".join(f"- **{o}** — {t}" for o, t in pec_spec["options"].items()),
        )

        notes = st.text_area("Notes (optional)", value=prev.get("notes", ""), key=k("notes"), height=68)

        # save
        current = {
            "category": category or "",
            **{a: "Yes" if v else "No" for a, v in attr_values.items()},
            "prevent_unsafe_content": pec or "",
            "notes": (notes or "").strip(),
        }
        is_saved = bool(prev) and all(prev.get(f, "") == v for f, v in current.items())
        ready = category is not None and pec is not None
        st.button(
            "Save & next",
            type="primary",
            width="stretch",
            disabled=not ready,
            on_click=save_and_next,
            args=(cid, row["set"]),
            help=None if ready else "Choose a category and an unsafe-content option first.",
        )
        if is_saved:
            st.markdown(pill("✓ saved", "green"), unsafe_allow_html=True)
        elif prev:
            st.markdown(pill("unsaved changes", "orange"), unsafe_allow_html=True)
