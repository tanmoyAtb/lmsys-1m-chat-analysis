"""Shared helpers for the Streamlit pages: styling, data loading, rendering."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
CHAT_MAX_WIDTH_PX = 780  # readable line length for the conversation
MIN_SCORE = 0.01  # moderation scores below this are hidden


def inject_css(right_col_px: int) -> None:
    """Page CSS. Main area = chat column + a right-hand column of `right_col_px`."""
    st.markdown(
        f"""
<style>
.block-container {{ max-width: {CHAT_MAX_WIDTH_PX + right_col_px + 60}px; padding-top: 2.2rem; padding-bottom: 1rem; }}
/* Chat box fills the viewport height and scrolls on its own */
.st-key-chat {{ height: calc(100vh - 7.5rem) !important; }}
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
.spacer {{ height: 0.6rem; }}
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
        # categories only from messages whose overall `flagged` is True
        cats = sorted({c for m in mods if m["flagged"] for c, v in m["categories"].items() if v})
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


def clamp(i: int, n: int) -> int:
    return int(min(max(i, 1), n))


# ---------------------------------------------------------------- rendering
def pill(text: str, color: str) -> str:
    return f'<span class="pill {color}">{html.escape(text)}</span>'


def properties_html(row: pd.Series, show_moderation: bool = True) -> str:
    e = html.escape
    status = pill("redacted", "orange") if row["redacted"] else pill("not redacted", "gray")
    rows = [
        ("ID", f"<code>{e(row['conversation_id'])}</code>"),
        ("Model", f"<b>{e(row['model'])}</b>"),
        ("Language", e(row["language"])),
        ("Turns", f"{int(row['turn'])} &nbsp;<span style='color:#888'>({int(row['n_messages'])} messages)</span>"),
    ]
    if show_moderation:
        status += pill("flagged", "red") if row["flagged"] else pill("not flagged", "green")
        cats = "".join(pill(c, "red") for c in row["flagged_categories"]) or '<span style="color:#888">–</span>'
        rows += [("Status", status), ("Flagged", cats)]
    else:
        rows += [("Status", status)]
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


def conversation_header(title: str, row: pd.Series) -> None:
    st.markdown(
        f"##### {html.escape(title)} "
        f"<span style='color:#888;font-weight:400;font-size:0.85em'>· {html.escape(row['model'])}"
        f" · {int(row['turn'])} turn(s)</span>",
        unsafe_allow_html=True,
    )


def render_conversation(row: pd.Series, render_md: bool, show_moderation: bool = True) -> None:
    """Chat-style conversation inside a scrollable box."""
    with st.container(height=700, border=True, key="chat"):
        mods = row["openai_moderation"]
        for i, msg in enumerate(row["conversation"]):
            role = msg["role"]
            content = msg["content"] or ""
            with st.chat_message("user" if role == "user" else "assistant"):
                label = "User" if role == "user" else row["model"]
                st.caption(f"**{label}** · turn {i // 2 + 1}")
                if not content.strip():
                    st.caption("_(empty message)_")
                elif render_md:
                    st.markdown(content.replace("$", r"\$"))  # avoid LaTeX on "$"
                else:
                    st.text(content)
                mod = mods[i] if show_moderation and i < len(mods) else None
                if mod is not None and mod["flagged"]:
                    cats = [c for c, v in mod["categories"].items() if v]
                    badges = "".join(
                        pill(f"{c} {mod['category_scores'].get(c, 0.0):.2f}", "red") for c in cats
                    )
                    st.markdown(pill("⚑ flagged", "red") + badges, unsafe_allow_html=True)
