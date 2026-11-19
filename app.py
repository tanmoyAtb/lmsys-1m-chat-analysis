"""Conversation explorer for the LMSYS-Chat-1M sample.

Run:  uv run streamlit run app.py

Layout:
  - Sidebar (fixed): sample picker, navigation, and the conversation's properties.
  - Main: the conversation rendered chat-style in a scrollable box.

Font sizes are set in .streamlit/config.toml.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

SAMPLE_DIR = Path(__file__).resolve().parent / "data" / "sample"
CHAT_MAX_WIDTH_PX = 780  # readable line length for the conversation
NAV_WIDTH_PX = 170  # navigation column to the right of the conversation
MIN_SCORE = 0.01  # moderation scores below this are hidden in the sidebar

st.set_page_config(page_title="LMSYS-Chat-1M Explorer", page_icon="💬", layout="wide")

st.markdown(
    f"""
<style>
/* Main column: narrow, centered, less top padding */
.block-container {{ max-width: {CHAT_MAX_WIDTH_PX + NAV_WIDTH_PX + 60}px; padding-top: 2.2rem; padding-bottom: 1rem; }}
/* Chat box fills the viewport height and scrolls on its own */
.st-key-chat {{ height: calc(100vh - 7.5rem) !important; }}
/* Tighter chat bubbles */
[data-testid="stChatMessage"] {{ padding: 0.5rem 0.75rem; gap: 0.6rem; }}
[data-testid="stChatMessage"] p {{ line-height: 1.5; }}
/* Compact sidebar */
[data-testid="stSidebarUserContent"] {{ padding-top: 0.5rem; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.5rem; }}
/* Property table */
table.props {{ width: 100%; border-collapse: collapse; font-size: 1rem; margin: 0; }}
table.props td {{ padding: 3px 4px; border: none; vertical-align: top; }}
table.props td.k {{ color: #888; white-space: nowrap; width: 1%; padding-right: 10px; }}
table.props code {{ font-size: 0.8rem; word-break: break-all; }}
.pill {{ display: inline-block; padding: 0 7px; border-radius: 8px; font-size: 0.85rem; margin: 1px 2px 1px 0; }}
.pill.red {{ background: rgba(255,75,75,.15); color: #e03c3c; }}
.pill.orange {{ background: rgba(255,164,33,.18); color: #c77700; }}
.pill.green {{ background: rgba(33,195,84,.15); color: #1a9c44; }}
.pill.gray {{ background: rgba(128,128,128,.15); color: #888; }}
/* Moderation score bars */
.scores {{ font-size: 0.9rem; }}
.scores .r {{ display: flex; align-items: center; gap: 6px; line-height: 1.5; }}
.scores .n {{ width: 10.5rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #888; }}
.scores .n.hit {{ color: #e03c3c; font-weight: 600; }}
.scores .b {{ flex: 1; height: 6px; background: rgba(128,128,128,.15); border-radius: 3px; overflow: hidden; }}
.scores .b > div {{ height: 100%; background: #8a8a8a; }}
.scores .b > div.hit {{ background: #e03c3c; }}
.scores .v {{ width: 2.6rem; text-align: right; font-variant-numeric: tabular-nums; color: #888; }}
.section {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: .05em; color: #888; margin: 0.6rem 0 0.2rem; }}
.empty {{ color: #888; font-size: 0.9rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading sample…")
def load_sample(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    def conv_flags(mods) -> tuple[bool, list[str], dict[str, float]]:
        flagged = any(m["flagged"] for m in mods)
        cats = sorted({c for m in mods for c, v in m["categories"].items() if v})
        max_scores: dict[str, float] = {}
        for m in mods:
            for c, s in m["category_scores"].items():
                max_scores[c] = max(max_scores.get(c, 0.0), float(s or 0.0))
        return flagged, cats, max_scores

    info = df["openai_moderation"].map(conv_flags)
    df["flagged"] = info.map(lambda x: x[0])
    df["flagged_categories"] = info.map(lambda x: x[1])
    df["max_scores"] = info.map(lambda x: x[2])
    df["n_messages"] = df["conversation"].map(len)
    return df.reset_index(drop=True)


samples = sorted(SAMPLE_DIR.glob("*.parquet"))
if not samples:
    st.error(f"No sample found in `{SAMPLE_DIR}`. Run `uv run python scripts/make_sample.py` first.")
    st.stop()


# ---------------------------------------------------------------- helpers
def _clamp(i: int, n: int) -> int:
    return int(min(max(i, 1), n))


def go(delta: int, n: int) -> None:
    st.session_state.conv_no = _clamp(st.session_state.conv_no + delta, n)


def go_random(n: int) -> None:
    st.session_state.conv_no = int(np.random.default_rng().integers(1, n + 1))


def pill(text: str, color: str) -> str:
    return f'<span class="pill {color}">{html.escape(text)}</span>'


def properties_html(row: pd.Series) -> str:
    e = html.escape
    status = (pill("redacted", "orange") if row["redacted"] else pill("not redacted", "gray")) + (
        pill("flagged", "red") if row["flagged"] else pill("not flagged", "green")
    )
    cats = "".join(pill(c, "red") for c in row["flagged_categories"]) or '<span style="color:#888">–</span>'
    rows = [
        ("ID", f"<code>{e(row['conversation_id'])}</code>"),
        ("Model", f"<b>{e(row['model'])}</b>"),
        ("Language", e(row["language"])),
        ("Turns", f"{int(row['turn'])} &nbsp;<span style='color:#888'>({int(row['n_messages'])} messages)</span>"),
        ("Status", status),
        ("Flagged", cats),
    ]
    body = "".join(f'<tr><td class="k">{k}</td><td>{v}</td></tr>' for k, v in rows)
    return f'<table class="props">{body}</table>'


def scores_html(row: pd.Series) -> str:
    flagged = set(row["flagged_categories"])
    items = sorted(
        ((c, s) for c, s in row["max_scores"].items() if s >= MIN_SCORE or c in flagged),
        key=lambda kv: -kv[1],
    )
    if not items:
        return f'<div class="empty">No category scored ≥ {MIN_SCORE}</div>'
    out = []
    for cat, s in items:
        hit = " hit" if cat in flagged else ""
        out.append(
            f'<div class="r"><span class="n{hit}" title="{html.escape(cat)}">{html.escape(cat)}</span>'
            f'<span class="b"><div class="{hit.strip()}" style="width:{s * 100:.1f}%"></div></span>'
            f'<span class="v">{s:.2f}</span></div>'
        )
    return f'<div class="scores">{"".join(out)}</div>'


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    sample_path = st.selectbox(
        "Sample", samples, format_func=lambda p: p.stem, label_visibility="collapsed"
    )
    df = load_sample(str(sample_path))
    n = len(df)

    if "conv_no" not in st.session_state:
        st.session_state.conv_no = 1
    st.session_state.conv_no = _clamp(st.session_state.conv_no, n)

    row = df.iloc[st.session_state.conv_no - 1]

    st.markdown('<div class="section">Properties</div>' + properties_html(row), unsafe_allow_html=True)
    st.markdown(
        '<div class="section">Moderation · max score over messages</div>' + scores_html(row),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)  # spacing
    render_md = st.toggle("Render markdown", value=True, help="Off shows the raw text exactly as stored.")


# ---------------------------------------------------------------- main: conversation + navigation
col_chat, col_nav = st.columns([CHAT_MAX_WIDTH_PX, NAV_WIDTH_PX], gap="medium")

with col_nav:
    st.markdown("<div style='height:2.1rem'></div>", unsafe_allow_html=True)  # align with chat box
    st.number_input(f"Conversation (1–{n:,})", min_value=1, max_value=n, step=1, key="conv_no")
    c_prev, c_next = st.columns(2, gap="small")
    c_prev.button("◀ Prev", on_click=go, args=(-1, n), width="stretch")
    c_next.button("Next ▶", on_click=go, args=(1, n), width="stretch")
    st.button("🎲 Random", on_click=go_random, args=(n,), width="stretch")

with col_chat:
    st.markdown(
        f"##### Conversation {st.session_state.conv_no:,} / {n:,} "
        f"<span style='color:#888;font-weight:400;font-size:0.85em'>· {html.escape(row['model'])}"
        f" · {int(row['turn'])} turn(s)</span>",
        unsafe_allow_html=True,
    )
    chat = st.container(height=700, border=True, key="chat")

with chat:
    mods = row["openai_moderation"]
    for i, msg in enumerate(row["conversation"]):
        role = msg["role"]
        content = msg["content"] or ""
        mod = mods[i] if i < len(mods) else None
        with st.chat_message("user" if role == "user" else "assistant"):
            label = "User" if role == "user" else row["model"]
            st.caption(f"**{label}** · turn {i // 2 + 1}")
            if not content.strip():
                st.caption("_(empty message)_")
            elif render_md:
                # Escape $ so prices etc. aren't rendered as LaTeX.
                st.markdown(content.replace("$", r"\$"))
            else:
                st.text(content)
            if mod is not None and mod["flagged"]:
                cats = [c for c, v in mod["categories"].items() if v]
                badges = "".join(pill(f"{c} {mod['category_scores'].get(c, 0.0):.2f}", "red") for c in cats)
                st.markdown(pill("⚑ flagged", "red") + badges, unsafe_allow_html=True)
