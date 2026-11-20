"""LMSYS-Chat-1M analysis UI.

Run:  uv run streamlit run app.py

Pages:
  - Explorer: browse any sample, with moderation info.
  - Annotate: human labeling of the 100 annotation conversations (moderation hidden).

Font sizes are set in .streamlit/config.toml.
"""

import streamlit as st

st.set_page_config(page_title="LMSYS-Chat-1M", page_icon="💬", layout="wide")

nav = st.navigation(
    [
        st.Page("views/explorer.py", title="Explorer", icon="💬", default=True),
        st.Page("views/annotate.py", title="Annotate", icon="✏️"),
    ],
    position="top",
)
nav.run()
