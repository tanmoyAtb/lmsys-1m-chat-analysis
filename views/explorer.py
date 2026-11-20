"""Explorer page: browse any sample with moderation info visible."""

from __future__ import annotations

import numpy as np
import streamlit as st

from ui.common import (
    CHAT_MAX_WIDTH_PX,
    SAMPLE_DIR,
    clamp,
    conversation_header,
    inject_css,
    load_sample,
    properties_html,
    render_conversation,
    scores_html,
)

NAV_WIDTH_PX = 170

inject_css(NAV_WIDTH_PX)

samples = sorted(SAMPLE_DIR.glob("*.parquet"))
if not samples:
    st.error(f"No sample found in `{SAMPLE_DIR}`. Run `uv run python scripts/make_sample.py` first.")
    st.stop()


def go(delta: int, n: int) -> None:
    st.session_state.conv_no = clamp(st.session_state.conv_no + delta, n)


def go_random(n: int) -> None:
    st.session_state.conv_no = int(np.random.default_rng().integers(1, n + 1))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    sample_path = st.selectbox("Sample", samples, format_func=lambda p: p.stem, label_visibility="collapsed")
    df = load_sample(str(sample_path))
    n = len(df)

    if "conv_no" not in st.session_state:
        st.session_state.conv_no = 1
    st.session_state.conv_no = clamp(st.session_state.conv_no, n)
    row = df.iloc[st.session_state.conv_no - 1]

    st.markdown('<div class="section">Properties</div>' + properties_html(row), unsafe_allow_html=True)
    st.markdown(
        '<div class="section">Moderation · max score over messages</div>' + scores_html(row),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    render_md = st.toggle("Render markdown", value=True, help="Off shows the raw text exactly as stored.")

# ---------------------------------------------------------------- main
col_chat, col_nav = st.columns([CHAT_MAX_WIDTH_PX, NAV_WIDTH_PX], gap="medium")

with col_nav:
    st.markdown("<div style='height:2.1rem'></div>", unsafe_allow_html=True)  # align with chat box
    st.number_input(f"Conversation (1–{n:,})", min_value=1, max_value=n, step=1, key="conv_no")
    c_prev, c_next = st.columns(2, gap="small")
    c_prev.button("◀ Prev", on_click=go, args=(-1, n), width="stretch")
    c_next.button("Next ▶", on_click=go, args=(1, n), width="stretch")
    st.button("🎲 Random", on_click=go_random, args=(n,), width="stretch")

with col_chat:
    conversation_header(f"Conversation {st.session_state.conv_no:,} / {n:,}", row)
    render_conversation(row, render_md, show_moderation=True)
