"""Build REPORT.md and the charts in reports/ from the committed data.

Every number in REPORT.md is computed here and filled into reports/REPORT.template.md
(placeholders look like ${name}). Nothing is typed by hand.

Inputs (all committed):
  data/sample/sample_english_5000_seed42.parquet   model, turns, OpenAI moderation
  annotation/human_labels.csv                      human labels (100)
  annotation/llm_labels_sample5000_<model>.csv     LLM labels (5,000)
  analysis/clustering/{prompts_A,conversations_B}.csv, mapping.yaml, clusters_*.csv
  reference/paper_fig3_topics.csv                  paper Figure 3

Usage:
  uv run python scripts/make_reports.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from string import Template

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import matplotlib.ticker  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
TEMPLATE = REPORTS / "REPORT.template.md"
OUT = ROOT / "REPORT.md"

LLM_CSV = sorted((ROOT / "annotation").glob("llm_labels_sample5000_*.csv"))[0]
MIN_MODEL_N = 20  # min conversations (or applicable conversations) to show a model's rate

# ---------------------------------------------------------------- style (dataviz reference palette, light)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SOURCE_COLOR = {"Paper": SLOT[0], "Clustering": SLOT[1], "LLM": SLOT[2], "Human": SLOT[3]}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 150,
})

SHORT = {
    "Technical and Software-related": "Technical & Software",
    "Cultural, Social, and Geographical": "Cultural, Social & Geo",
    "Language and Content Creation": "Language & Content",
    "Business and Specific Inquiries": "Business & Specific",
    "Explicit Content": "Explicit Content",
    "Others": "Others",
}
ATTR_TITLE = {"anthropomorphism": "Anthropomorphism", "sycophancy": "Sycophancy",
              "fear_inducing": "Fear-inducing", "harmful": "Harmful"}


# ---------------------------------------------------------------- helpers
def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v)
                                       for v in row) + " |")
    return "\n".join(lines)


def shares(s: pd.Series, cats: list[str]) -> pd.Series:
    return s.value_counts(normalize=True).reindex(cats).fillna(0.0)


def savefig(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    return f"reports/figures/{name}"


def kappa_row(a: pd.Series, b: pd.Series) -> dict:
    return {"n": len(a), "agreement": pct((a.values == b.values).mean(), 0),
            "kappa": f"{cohen_kappa_score(a, b):.2f}"}


def grouped_hbar(ax, data: pd.DataFrame, colors: list[str], title: str, xmax: float,
                 values: bool = False, legend_loc: str = "lower right") -> None:
    """data: rows = categories (top to bottom), columns = series (percent values).
    values=True writes each bar's percentage at its end."""
    n_series = data.shape[1]
    h = 0.8 / n_series
    y = np.arange(len(data))[::-1]
    for j, (col, color) in enumerate(zip(data.columns, colors)):
        yy = y + (n_series - 1 - j - (n_series - 1) / 2) * h
        ax.barh(yy, data[col].values, height=h * 0.92, color=color, edgecolor=SURFACE, linewidth=1, label=col)
        if values:
            for yi, v in zip(yy, data[col].values):  # zeros labeled too, so an empty row isn't read as missing
                ax.text(v + xmax * 0.008, yi, f"{v:.1f}%", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [SHORT.get(c, c) for c in data.index])
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.set_title(title)
    ax.legend(loc=legend_loc, fontsize=8.5)


# ---------------------------------------------------------------- load
def load() -> dict:
    spec = yaml.safe_load((ROOT / "annotation" / "labels.yaml").read_text())
    cats = [o["name"] for o in spec["category"]["options"]]
    attrs = list(spec["attributes"])

    sample = pd.read_parquet(ROOT / "data/sample/sample_english_5000_seed42.parquet",
                             columns=["conversation_id", "model", "turn", "openai_moderation"])
    sample["flagged"] = sample["openai_moderation"].map(lambda ms: any(m["flagged"] for m in ms))
    sample = sample.drop(columns="openai_moderation").set_index("conversation_id")

    llm = pd.read_csv(LLM_CSV, dtype=str, keep_default_na=False).set_index("conversation_id")
    human = pd.read_csv(ROOT / "annotation/human_labels.csv", dtype=str, keep_default_na=False)
    human = human.set_index("conversation_id")

    mapping = yaml.safe_load((ROOT / "analysis/clustering/mapping.yaml").read_text())
    pa = pd.read_csv(ROOT / "analysis/clustering/prompts_A.csv")
    pa["category"] = pa["cluster"].map({c: v["category"] for c, v in mapping["A"].items()})
    cb = pd.read_csv(ROOT / "analysis/clustering/conversations_B.csv").set_index("conversation_id")
    cb["category"] = cb["cluster"].map({c: v["category"] for c, v in mapping["B"].items()})
    ca = pd.read_csv(ROOT / "analysis/clustering/clusters_A.csv").set_index("cluster")
    cbs = pd.read_csv(ROOT / "analysis/clustering/clusters_B.csv").set_index("cluster")
    paper = pd.read_csv(ROOT / "reference/paper_fig3_topics.csv")
    # Which paper topic each of our clusters is (LLM-proposed, reviewed): int or None
    pt = yaml.safe_load((ROOT / "analysis/clustering/paper_topics.yaml").read_text())
    ptopic = {v: {c: (None if str(x["paper_topic"]).lower() == "none" else int(x["paper_topic"]))
                  for c, x in pt[v].items()} for v in ("A", "B")}

    return dict(spec=spec, cats=cats, attrs=attrs, sample=sample, llm=llm, human=human,
                pa=pa, cb=cb, ca=ca, cbs=cbs, paper=paper, mapping=mapping, ptopic=ptopic)


def display_name(D: dict, v: str, c: int) -> str:
    """Paper topic name if the cluster is the same topic as a paper cluster, else our own k-means name."""
    p = D["ptopic"][v][c]
    title = (D["ca"] if v == "A" else D["cbs"]).loc[c, "title"]
    return D["paper"].set_index("cluster").loc[p, "label"] if p else title


# ---------------------------------------------------------------- sections
def section_overview(D: dict, V: dict) -> None:
    s, pa, cb, h = D["sample"], D["pa"], D["cb"], D["human"]
    V.update(
        n_sample=f"{len(s):,}", n_prompts_kept=f"{len(pa):,}", n_conv_b=f"{len(cb):,}",
        n_conv_dropped=f"{len(s) - len(cb):,}", pct_conv_dropped=pct((len(s) - len(cb)) / len(s)),
        n_models=s["model"].nunique(), llm_model=D["llm"]["annotator"].iloc[0].removeprefix("llm:"),
        labels_version=D["llm"]["labels_version"].iloc[0],
        n_h_random=int((h["set"] == "random60").sum()), n_h_flagged=int((h["set"] == "flagged40").sum()),
        n_h_random_b=int(((h["set"] == "random60") & h.index.isin(cb.index)).sum()),
        n_h_b=int(h.index.isin(cb.index).sum()),
        single_turn=pct((s["turn"] == 1).mean()), flagged_5000=pct(s["flagged"].mean()),
    )
    dropped = D["llm"].loc[~D["llm"].index.isin(cb.index), "category"]
    t = dropped.value_counts().reindex(D["cats"]).fillna(0).astype(int).rename("conversations")
    t.index = [SHORT[c] for c in t.index]
    V["table_dropped"] = md_table(t.to_frame().rename_axis("LLM category"))
    V["dropped_others_lang"] = pct((dropped.isin(["Others", "Language and Content Creation"])).mean(), 0)


def section_topics(D: dict, V: dict) -> None:
    """Prompt-level vs conversation-level: matched topics and category shares."""
    paper, ca, cbs = D["paper"], D["ca"], D["cbs"]
    pp = paper.set_index("cluster")
    pA, pB = D["ptopic"]["A"], D["ptopic"]["B"]
    rows, vals = [], []
    for n, r in pp.iterrows():
        a_ids = [c for c, p in pA.items() if p == n]
        b_ids = [c for c, p in pB.items() if p == n]
        f = lambda ids, s: f"{s.loc[ids].sum():.1f}% ({', '.join(map(str, ids))})" if ids else "–"  # noqa: E731
        rows.append({"Paper topic": f"{n}. {r['label']}", "Paper (prompts)": f"{r['percent']:.1f}%",
                     "Prompt clustering": f(a_ids, ca["percent"]), "Conversation clustering": f(b_ids, cbs["percent"])})
        vals.append((f"{n}. {r['label']}", r["percent"], ca["percent"].loc[a_ids].sum() if a_ids else np.nan,
                     cbs["percent"].loc[b_ids].sum() if b_ids else np.nan))
    a_new = [c for c, p in pA.items() if p is None]
    b_new = [c for c, p in pB.items() if p is None]
    rows.append({"Paper topic": "**Not in the paper** (our own topics, below)", "Paper (prompts)": "–",
                 "Prompt clustering": f"{ca['percent'].loc[a_new].sum():.1f}%",
                 "Conversation clustering": f"{cbs['percent'].loc[b_new].sum():.1f}%"})
    vals.append(("Not in the paper (all other clusters)", np.nan, ca["percent"].loc[a_new].sum(),
                 cbs["percent"].loc[b_new].sum()))
    V["table_matched_topics"] = md_table(pd.DataFrame(rows), index=False)
    new_rows = ([{"Approach": "Prompt clustering", "cluster": c, "topic (our name)": ca.loc[c, "title"],
                  "share": f"{ca.loc[c, 'percent']:.1f}%"} for c in a_new]
                + [{"Approach": "Conversation clustering", "cluster": c, "topic (our name)": cbs.loc[c, "title"],
                    "share": f"{cbs.loc[c, 'percent']:.1f}%"} for c in b_new])
    V["table_new_topics"] = md_table(pd.DataFrame(new_rows), index=False)
    V["n_paper_unmatched_A"] = int(sum(np.isnan(v[2]) for v in vals[:-1]))
    V["n_paper_unmatched_B"] = int(sum(np.isnan(v[3]) for v in vals[:-1]))

    # Small multiples: one panel per approach, the paper's 20 topics + "not in the paper", single color
    names = [v[0] for v in vals]
    panels = [("Paper\n(prompts, 100K conv.)", [v[1] for v in vals]),
              ("Prompt clustering", [v[2] for v in vals]),
              ("Conversation clustering", [v[3] for v in vals])]
    xmax = np.nanmax([x for _, xs in panels for x in xs[:-1]]) * 1.3  # scale to the topics only
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 7.2), sharey=True)
    y = np.arange(len(vals))[::-1].astype(float)
    y[-1] -= 0.6  # separate the "not in the paper" row
    for ax, (title, xs) in zip(axes, panels):
        xs = np.array(xs, dtype=float)
        ax.barh(y[:-1], np.nan_to_num(xs[:-1]), height=0.66, color=SLOT[0], edgecolor=SURFACE, linewidth=1)
        for yi, x in zip(y[:-1], xs[:-1]):
            ax.text((0 if np.isnan(x) else x) + xmax * 0.025, yi, "–" if np.isnan(x) else f"{x:.1f}%",
                    va="center", fontsize=7.5, color=INK2)
        last = xs[-1]  # "not in the paper": a number only, so the axis stays readable
        ax.text(xmax * 0.025, y[-1], "–" if np.isnan(last) else f"{last:.1f}% of all", va="center",
                fontsize=8.5, color=INK, fontweight="bold")
        ax.set_ylim(y[-1] - 0.9, y[0] + 0.7)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=10, loc="center")
    axes[0].set_yticks(y, names, fontsize=8)
    fig.suptitle("The paper's 20 topics in each approach (share of prompts / conversations)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.text(0.01, -0.005, "Our clusters carry a paper topic's name where they are the same topic "
             "(LLM-proposed, reviewed; analysis/clustering/paper_topics.yaml).  – = no matching cluster.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    V["fig_topics_matched"] = savefig(fig, "00b_matched_topics.png")

    # Cluster lists with mapped category
    for v, cs in [("A", ca), ("B", cbs)]:
        t = pd.DataFrame({
            "percent": cs["percent"].map(lambda x: f"{x:.1f}%"),
            "topic": [display_name(D, v, c) for c in cs.index],
            "paper topic": [D["ptopic"][v][c] or "new" for c in cs.index],
            "k-means name": cs["title"],
            "category": [SHORT[D["mapping"][v][c]["category"]] for c in cs.index]}, index=cs.index)
        V[f"table_clusters_{v}"] = md_table(t.rename_axis("cluster"))

    # Category shares at the two units
    cats = D["cats"]
    pcat = paper.groupby("category")["percent"].sum().reindex(cats).fillna(0) / 100
    a = shares(D["pa"]["category"], cats)
    b = shares(D["cb"]["category"], cats)
    llm_b = D["llm"].loc[D["cb"].index, "category"]
    llm_conv = shares(llm_b, cats)
    # LLM labels counted once per kept prompt (the paper's unit)
    w = D["cb"]["n_prompts"]
    llm_prompt = (w.groupby(llm_b).sum() / w.sum()).reindex(cats).fillna(0)
    LLM_W = "LLM, prompt-weighted"
    t = pd.DataFrame({"Paper (prompts)": pcat, "Prompt clustering": a, LLM_W: llm_prompt,
                      "Conversation clustering": b, "LLM per conversation": llm_conv})
    V["table_unit"] = md_table((t.map(lambda x: pct(x)))
                               .set_axis([SHORT[c] for c in t.index]).rename_axis("Category"))
    V.update(
        unit_expl_paper=pct(pcat["Explicit Content"]), unit_expl_llm_prompt=pct(llm_prompt["Explicit Content"]),
        unit_expl_llm_conv=pct(llm_conv["Explicit Content"]),
        unit_lang_a=pct(a["Language and Content Creation"]), unit_lang_b=pct(b["Language and Content Creation"]),
        unit_tech_a=pct(a["Technical and Software-related"]), unit_tech_b=pct(b["Technical and Software-related"]),
        unit_bus_a=pct(a["Business and Specific Inquiries"]), unit_bus_b=pct(b["Business and Specific Inquiries"]),
        unit_cult_a=pct(a["Cultural, Social, and Geographical"]),
        unit_cult_b=pct(b["Cultural, Social, and Geographical"]),
    )
    # prompts per conversation by LLM category (why units differ)
    ppc = w.groupby(llm_b).mean().reindex(cats)
    V["table_prompts_per_conv"] = md_table(
        ppc.dropna().map(lambda x: f"{x:.2f}").rename("kept prompts per conversation")
        .set_axis([SHORT[c] for c in ppc.dropna().index]).rename_axis("LLM category").to_frame())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    grouped_hbar(axes[0], t[["Paper (prompts)", "Prompt clustering", LLM_W]] * 100,
                 [SOURCE_COLOR["Paper"], SOURCE_COLOR["Clustering"], SOURCE_COLOR["LLM"]],
                 "Counted per prompt", 55, values=True)
    grouped_hbar(axes[1], t[["Conversation clustering", "LLM per conversation"]] * 100,
                 [SOURCE_COLOR["Clustering"], SOURCE_COLOR["LLM"]], "Counted per conversation", 55, values=True)
    fig.suptitle("Category shares: prompt vs conversation as the unit", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    V["fig_unit"] = savefig(fig, "01_unit_prompt_vs_conversation.png")


CAT_COLOR_SLOTS = [0, 1, 2, 3, 4, 5]  # category order from labels.yaml -> palette slot
CAT_TAG = {"Technical and Software-related": "Tech", "Cultural, Social, and Geographical": "Cult",
           "Language and Content Creation": "Lang", "Business and Specific Inquiries": "Bus",
           "Explicit Content": "Expl", "Others": "Oth"}


def section_topics20(D: dict, V: dict) -> None:
    """The 20 topics of the paper, prompt clustering and conversation clustering side by side (like the paper's Fig. 3)."""
    cats = D["cats"]
    color = {c: SLOT[i] for c, i in zip(cats, CAT_COLOR_SLOTS)}
    paper = D["paper"].set_index("cluster")

    def names_for(v: str) -> pd.Series:
        """Paper topic name with its number for matched clusters; own name marked 'new' otherwise."""
        idx = (D["ca"] if v == "A" else D["cbs"]).index
        return pd.Series({c: (f"{display_name(D, v, c)} (=P{D['ptopic'][v][c]})" if D["ptopic"][v][c]
                              else f"{display_name(D, v, c)} (new)") for c in idx})

    panels = [
        ("Paper: 20 clusters of prompts (100K conversations)", paper["label"], paper["percent"],
         paper["category"]),
        (f"Prompt clustering: 20 clusters of prompts (n={len(D['pa']):,})", names_for("A"), D["ca"]["percent"],
         pd.Series({c: v["category"] for c, v in D["mapping"]["A"].items()})),
        (f"Conversation clustering: 20 clusters of conversations (n={len(D['cb']):,})", names_for("B"),
         D["cbs"]["percent"], pd.Series({c: v["category"] for c, v in D["mapping"]["B"].items()})),
    ]
    xmax = max(p[2].max() for p in panels) * 1.18
    fig, axes = plt.subplots(3, 1, figsize=(10, 17))
    for ax, (title, labels, perc, catmap) in zip(axes, panels):
        order = perc.sort_values(ascending=False).index
        y = np.arange(len(order))[::-1]
        ax.barh(y, perc.loc[order].values, height=0.72,
                color=[color[catmap.loc[c]] for c in order], edgecolor=SURFACE, linewidth=1)
        ax.set_yticks(y, [f"{c}. {labels.loc[c]}  [{CAT_TAG[catmap.loc[c]]}]" for c in order], fontsize=8.5)
        for yi, v in zip(y, perc.loc[order].values):
            ax.text(v + xmax * 0.008, yi, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0)
        ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[c]) for c in cats if c != "Others"]
    fig.legend(handles, [f"{SHORT[c]} [{CAT_TAG[c]}]" for c in cats if c != "Others"], loc="upper center",
               ncol=3, bbox_to_anchor=(0.55, 1.0), fontsize=9)
    fig.suptitle("Topic distribution: paper vs prompt clustering vs conversation clustering", x=0.01,
                 ha="left", y=1.025, fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    V["fig_topics20"] = savefig(fig, "00_topics_20_clusters.png")

    t = pd.DataFrame({"cluster": paper.index, "label": paper["label"].values,
                      "category": [SHORT[c] for c in paper["category"]],
                      "percent": paper["percent"].map(lambda x: f"{x:.2f}%").values})
    V["table_clusters_paper"] = md_table(t, index=False)


def section_categories(D: dict, V: dict) -> None:
    cats, cb, llm, h = D["cats"], D["cb"], D["llm"], D["human"]
    hb = h[h.index.isin(cb.index)]
    hr = hb[hb["set"] == "random60"]
    t = pd.DataFrame({
        "Paper (prompts)": D["paper"].groupby("category")["percent"].sum().reindex(cats).fillna(0) / 100,
        "Conversation clustering": shares(cb["category"], cats),
        "LLM": shares(llm.loc[cb.index, "category"], cats),
        "Human (random)": shares(hr["category"], cats),
        "LLM, all 5,000": shares(llm["category"], cats),
        "Human, all random 60": shares(h.loc[h["set"] == "random60", "category"], cats),
    })
    V["table_cat_pct"] = md_table(t.map(pct).set_axis([SHORT[c] for c in t.index]).rename_axis("Category"))
    V.update(cat_llm_bus=pct(t.loc["Business and Specific Inquiries", "LLM, all 5,000"]),
             cat_paper_bus=pct(t.loc["Business and Specific Inquiries", "Paper (prompts)"]),
             cat_llm_others=pct(t.loc["Others", "LLM, all 5,000"]),
             cat_h_others=pct(t.loc["Others", "Human, all random 60"]))

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    grouped_hbar(ax, t[["Conversation clustering", "LLM", "Human (random)"]] * 100,
                 [SOURCE_COLOR["Clustering"], SOURCE_COLOR["LLM"], SOURCE_COLOR["Human"]],
                 f"Category shares per conversation (the {len(cb):,} conversations that can be clustered)", 45, values=True)
    V["fig_cat_pct"] = savefig(fig, "02_category_shares.png")

    # Agreement (per conversation)
    hl = h.index  # all 100
    pairs = {
        "Human vs LLM": (h["category"], llm.loc[hl, "category"]),
        "Human vs Conversation clustering": (hb["category"], cb.loc[hb.index, "category"]),
        "LLM vs Conversation clustering": (llm.loc[cb.index, "category"], cb["category"]),
    }
    rows = {k: kappa_row(a, b) for k, (a, b) in pairs.items()}
    rows["Human vs Conversation clustering"]["n"] = len(hb)
    V["table_cat_agree"] = md_table(pd.DataFrame(rows).T.rename_axis("Pair"))
    for k, key in [("Human vs LLM", "hl"), ("Human vs Conversation clustering", "hb"), ("LLM vs Conversation clustering", "lb")]:
        V[f"kappa_{key}"] = rows[k]["kappa"]
        V[f"agree_{key}"] = rows[k]["agreement"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labs = [SHORT[c] for c in cats]
    for ax, (name, (a, b)) in zip(axes, pairs.items()):
        r, c = name.split(" vs ")
        cm = pd.crosstab(pd.Categorical(a, cats), pd.Categorical(b, cats), dropna=False)
        rown = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
        ax.imshow(rown.fillna(0).values, cmap=BLUES, vmin=0, vmax=1)
        ax.set_xticks(range(len(cats)), labs, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(cats)), labs if ax is axes[0] else [""] * len(cats), fontsize=8)
        ax.set_xlabel(c)
        ax.set_ylabel(r if ax is axes[0] else "")
        ax.grid(False)
        ax.set_title(f"{name}  (n={len(a):,})", fontsize=10)
        for i in range(len(cats)):
            for j in range(len(cats)):
                if cm.values[i, j]:
                    ax.text(j, i, cm.values[i, j], ha="center", va="center", fontsize=7.5,
                            color="white" if rown.values[i, j] > 0.55 else INK)
    fig.suptitle("Category confusion matrices (counts; shading = share of the row)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    V["fig_cat_confusion"] = savefig(fig, "03_category_confusion.png")

    # biggest confusions human vs LLM
    cm = pd.crosstab(h["category"], llm.loc[hl, "category"])
    off = [(r, c, cm.loc[r, c]) for r in cm.index for c in cm.columns if r != c and cm.loc[r, c] > 0]
    off.sort(key=lambda x: -x[2])
    V["top_confusions_hl"] = "; ".join(f"human {SHORT[r]} → LLM {SHORT[c]} ({n})" for r, c, n in off[:3])


def section_attributes(D: dict, V: dict) -> None:
    attrs, llm, h, s = D["attrs"], D["llm"], D["human"], D["sample"]
    hr = h[h["set"] == "random60"]
    rows = {}
    for a in attrs:
        rows[ATTR_TITLE[a]] = {
            "Human (random 60)": f"{(hr[a] == 'Yes').sum()} ({pct((hr[a] == 'Yes').mean())})",
            "LLM (same 60)": f"{(llm.loc[hr.index, a] == 'Yes').sum()} ({pct((llm.loc[hr.index, a] == 'Yes').mean())})",
            "LLM (5,000)": f"{(llm[a] == 'Yes').sum()} ({pct((llm[a] == 'Yes').mean())})",
            **{k: v for k, v in kappa_row(h[a], llm.loc[h.index, a]).items() if k != "n"},
            "Yes: human / LLM (100)": f"{(h[a] == 'Yes').sum()} / {(llm.loc[h.index, a] == 'Yes').sum()}",
        }
    V["table_attr"] = md_table(pd.DataFrame(rows).T.rename_axis("Attribute"))
    for a in attrs:
        V[f"attr_{a}_llm"] = pct((llm[a] == "Yes").mean())
        V[f"attr_{a}_kappa"] = f"{cohen_kappa_score(h[a], llm.loc[h.index, a]):.2f}"
    V["attr_any_llm"] = pct((llm[attrs] == "Yes").any(axis=1).mean())

    t = pd.DataFrame({"Human (random 60)": [(hr[a] == "Yes").mean() for a in attrs],
                      "LLM (5,000)": [(llm[a] == "Yes").mean() for a in attrs]},
                     index=[ATTR_TITLE[a] for a in attrs]) * 100
    fig, ax = plt.subplots(figsize=(7, 3.2))
    grouped_hbar(ax, t, [SOURCE_COLOR["Human"], SOURCE_COLOR["LLM"]], "Share of conversations with each attribute",
                 max(12, math.ceil(t.values.max() / 2) * 2 + 2), values=True)
    V["fig_attr"] = savefig(fig, "04_attribute_prevalence.png")

    # Same 100 conversations: human vs LLM (attributes + prevention)
    h100, l100 = h, llm.loc[h.index]
    ta = pd.DataFrame({"Human": [(h100[a] == "Yes").mean() for a in attrs],
                       "LLM": [(l100[a] == "Yes").mean() for a in attrs]},
                      index=[ATTR_TITLE[a] for a in attrs]) * 100
    opts = list(D["spec"]["prevent_unsafe_content"]["options"])
    tp = pd.DataFrame({"Human": h100["prevent_unsafe_content"].value_counts(normalize=True).reindex(opts).fillna(0),
                       "LLM": l100["prevent_unsafe_content"].value_counts(normalize=True).reindex(opts).fillna(0)}) * 100
    tp.index = [f"Prevented: {o}" if o in ("Yes", "No") else o for o in tp.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1]})
    grouped_hbar(axes[0], ta, [SOURCE_COLOR["Human"], SOURCE_COLOR["LLM"]], "Attributes: % Yes",
                 max(30, math.ceil(ta.values.max() / 5) * 5 + 5), values=True, legend_loc="upper right")
    grouped_hbar(axes[1], tp, [SOURCE_COLOR["Human"], SOURCE_COLOR["LLM"]], "Prevent unsafe content",
                 100, values=True, legend_loc="upper right")
    fig.suptitle(f"Human vs LLM on the same {len(h100)} conversations "
                 f"({(h100['set'] == 'random60').sum()} random + {(h100['set'] == 'flagged40').sum()} flagged)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    V["fig_attr_100"] = savefig(fig, "04a_human_vs_llm_same_100.png")
    t100 = pd.concat([ta.map(lambda x: f"{x:.0f}%"), tp.map(lambda x: f"{x:.0f}%")])
    V["table_attr_100"] = md_table(t100.rename_axis("Label (100 conversations)"))

    # by category (LLM, 5,000)
    byc = (llm[attrs] == "Yes").groupby(llm["category"]).mean().reindex(D["cats"])
    byc["n"] = llm["category"].value_counts().reindex(D["cats"])
    out = byc[attrs].map(pct).rename(columns=ATTR_TITLE)
    out.insert(0, "n", byc["n"].astype(int))
    V["table_attr_by_cat"] = md_table(out.set_axis([SHORT[c] for c in out.index]).rename_axis("LLM category"))

    # by model (LLM, 5,000)
    n = s["model"].value_counts()
    keep = n[n >= MIN_MODEL_N].index
    m = (llm[attrs] == "Yes").join(s["model"]).groupby("model").mean().loc[keep]
    m["n"] = n.loc[keep]
    m = m.sort_values("n", ascending=False)
    out = m[attrs].map(pct).rename(columns=ATTR_TITLE)
    out.insert(0, "n", m["n"].astype(int))
    V["table_attr_by_model"] = md_table(out.rename_axis("Model"))
    V["n_models_attr"] = len(keep)
    V["models_small"] = ", ".join(f"{k} ({v})" for k, v in n[n < MIN_MODEL_N].items()) or "none"
    highlights = []
    for a in attrs:
        top = m[a].sort_values(ascending=False).head(3)
        overall = (llm[a] == "Yes").mean()
        highlights.append(f"- **{ATTR_TITLE[a]}** (all models {pct(overall)}): "
                          + ", ".join(f"{k} {pct(v)} (n={int(m.loc[k, 'n'])})" for k, v in top.items()))
    V["attr_model_highlights"] = "\n".join(highlights)
    # Why llama-2-13b-chat scores high on anthropomorphism: what the LLM's rationales point to
    sel = llm[(s.loc[llm.index, "model"] == "llama-2-13b-chat") & (llm["anthropomorphism"] == "Yes")]
    feel = sel["notes"].str.contains(r"happy|glad|delight|pleas|excit|love|feel|caring", case=False)
    V["llama2_anthro_n"] = len(sel)
    V["llama2_anthro_feel"] = int(feel.sum())

    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(m) + 1.4))
    vals = m[attrs].values * 100
    vmax = max(10, np.nanmax(vals))
    im = ax.imshow(vals, cmap=BLUES, aspect="auto", vmin=0, vmax=vmax)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.1f}%", ha="center", va="center", fontsize=7.5,
                    color="white" if vals[i, j] > vmax * 0.55 else INK)
    ax.set_xticks(range(len(attrs)), [ATTR_TITLE[a] for a in attrs], rotation=20, ha="right")
    ax.set_yticks(range(len(m)), [f"{k}  (n={int(v)})" for k, v in m["n"].items()], fontsize=8.5)
    ax.grid(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("% of the model's conversations", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(f"Attributes by model (LLM labels; models with ≥{MIN_MODEL_N} conversations)")
    V["fig_attr_model"] = savefig(fig, "05_attributes_by_model.png")


def section_prevention(D: dict, V: dict) -> None:
    llm, s, h = D["llm"], D["sample"], D["human"]
    p = llm["prevent_unsafe_content"]
    app = p[p != "Not applicable"]
    ok = (app == "Yes")
    V.update(prev_n_app=f"{len(app):,}", prev_pct_app=pct(len(app) / len(llm)), prev_n_yes=f"{ok.sum():,}",
             prev_n_no=f"{(~ok).sum():,}", prev_rate=pct(ok.mean()))
    lo, hi = wilson(int(ok.sum()), len(app))
    V["prev_rate_ci"] = f"{pct(lo)}–{pct(hi)}"

    # human vs LLM agreement on prevention
    V["prev_kappa"] = f"{cohen_kappa_score(h['prevent_unsafe_content'], llm.loc[h.index, 'prevent_unsafe_content']):.2f}"
    V["prev_agree"] = pct((h["prevent_unsafe_content"] == llm.loc[h.index, "prevent_unsafe_content"]).mean(), 0)
    cm = pd.crosstab(h["prevent_unsafe_content"], llm.loc[h.index, "prevent_unsafe_content"])
    V["table_prev_cm"] = md_table(cm.rename_axis("Human \\ LLM"))

    # by model
    d = pd.DataFrame({"ok": ok}).join(s["model"])
    g = d.groupby("model")["ok"].agg(["sum", "count"])
    g = g[g["count"] >= MIN_MODEL_N].copy()
    g["rate"] = g["sum"] / g["count"]
    g[["lo", "hi"]] = [wilson(int(k), int(n)) for k, n in zip(g["sum"], g["count"])]
    g = g.sort_values("rate", ascending=False)
    small = d.groupby("model")["ok"].agg(["sum", "count"])
    small = small[small["count"] < MIN_MODEL_N].sort_values("count", ascending=False)
    out = pd.DataFrame({
        "applicable": g["count"].astype(int), "prevented": g["sum"].astype(int),
        "failed": (g["count"] - g["sum"]).astype(int), "prevention rate": g["rate"].map(pct),
        "95% CI": [f"{pct(a, 0)}–{pct(b, 0)}" for a, b in zip(g["lo"], g["hi"])]})
    V["table_prev_model"] = md_table(out.rename_axis("Model"))
    V["prev_models_small"] = ", ".join(f"{k} ({int(r['sum'])}/{int(r['count'])})" for k, r in small.iterrows())
    V["prev_best"] = ", ".join(f"{k} ({pct(r)})" for k, r in g["rate"].head(3).items())
    V["prev_worst"] = ", ".join(f"{k} ({pct(r)})" for k, r in g["rate"].tail(3)[::-1].items())
    V["n_models_prev"] = len(g)
    overall = ok.mean()
    above = g[g["lo"] > overall].index.tolist()
    below = g[g["hi"] < overall].index.tolist()
    V["prev_above"] = ", ".join(above) or "none"
    V["prev_below"] = ", ".join(below) or "none"
    V["prev_overlap"] = len(g) - len(above) - len(below)

    fig, ax = plt.subplots(figsize=(7, 0.36 * len(g) + 1.2))
    y = np.arange(len(g))[::-1]
    ax.hlines(y, g["lo"] * 100, g["hi"] * 100, color=GRID, linewidth=3)
    ax.plot(g["rate"] * 100, y, "o", color=SLOT[0], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for yi, r, hi in zip(y, g["rate"], g["hi"]):
        x = hi * 100 + 1.5
        ax.text(x if x < 92 else r * 100 - 1.5, yi + 0.28, f"{r * 100:.0f}%", va="center", fontsize=8,
                color=INK, ha="left" if x < 92 else "right")
    ax.axvline(ok.mean() * 100, color=INK2, linewidth=0.8)
    ax.text(ok.mean() * 100 + 1, -0.45, f"all models {pct(ok.mean(), 0)}", color=INK2, fontsize=8.5)
    ax.set_ylim(-0.7, len(g) - 0.4)
    ax.set_yticks(y, [f"{k}  ({int(n)})" for k, n in g["count"].items()])
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.set_title("Unsafe-content prevention rate by model (dot = rate, bar = 95% CI)")
    ax.set_xlabel(f"prevented ÷ conversations with an unsafe request (models with ≥{MIN_MODEL_N}; n in brackets)")
    V["fig_prev_model"] = savefig(fig, "06_prevention_by_model.png")

    # by request type (explicit vs other) and by turns
    cat = llm.loc[app.index, "category"]
    kind = np.where(cat == "Explicit Content", "Sexual (category Explicit)", "Other unsafe (toxic, harmful, …)")
    multi = np.where(s.loc[app.index, "turn"] > 1, "Multi-turn", "Single-turn")
    rows = []
    for name, grp in [("Request type", kind), ("Conversation length", multi)]:
        for k in sorted(set(grp)):
            sel = ok[grp == k]
            lo, hi = wilson(int(sel.sum()), len(sel))
            rows.append({"split": name, "group": k, "applicable": len(sel), "prevention rate": pct(sel.mean()),
                         "95% CI": f"{pct(lo, 0)}–{pct(hi, 0)}"})
    V["table_prev_split"] = md_table(pd.DataFrame(rows), index=False)
    V["prev_sexual"] = rows[[r["group"] for r in rows].index("Sexual (category Explicit)")]["prevention rate"]
    V["prev_other"] = rows[[r["group"] for r in rows].index("Other unsafe (toxic, harmful, …)")]["prevention rate"]
    V["prev_single"] = rows[[r["group"] for r in rows].index("Single-turn")]["prevention rate"]
    V["prev_multi"] = rows[[r["group"] for r in rows].index("Multi-turn")]["prevention rate"]


def section_moderation(D: dict, V: dict) -> None:
    llm, s = D["llm"], D["sample"]
    unsafe = llm["prevent_unsafe_content"] != "Not applicable"
    flagged = s.loc[llm.index, "flagged"]
    t = pd.crosstab(flagged.map({True: "OpenAI flagged", False: "Not flagged"}),
                    unsafe.map({True: "Unsafe request/content (LLM)", False: "Not applicable (LLM)"}))
    V["table_mod"] = md_table(t.rename_axis("OpenAI moderation \\ our label"))
    V["mod_unsafe_not_flagged"] = f"{int((unsafe & ~flagged).sum()):,}"
    V["mod_unsafe_not_flagged_pct"] = pct((~flagged[unsafe]).mean(), 0)
    nf = unsafe & ~flagged
    V["mod_nf_prevented"] = f"{int((llm.loc[nf, 'prevent_unsafe_content'] == 'Yes').sum()):,}"
    V["mod_nf_prevented_pct"] = pct((llm.loc[nf, "prevent_unsafe_content"] == "Yes").mean(), 0)
    V["mod_flagged_na"] = f"{int((~unsafe & flagged).sum()):,}"
    harm = llm["harmful"] == "Yes"
    V["mod_harm_not_flagged"] = pct((~flagged[harm]).mean(), 0)
    # failed prevention caught by moderation?
    failed = llm["prevent_unsafe_content"] == "No"
    V["mod_failed_flagged"] = pct(flagged[failed].mean(), 0)
    succ = llm["prevent_unsafe_content"] == "Yes"
    V["mod_success_flagged"] = pct(flagged[succ].mean(), 0)
    attrs_other = ["anthropomorphism", "sycophancy", "fear_inducing"]
    V["mod_attr_not_flagged"] = pct((~flagged[(llm[attrs_other] == "Yes").any(axis=1)]).mean(), 0)


def main() -> None:
    D = load()
    V: dict[str, object] = {}
    section_overview(D, V)
    section_topics(D, V)
    section_topics20(D, V)
    section_categories(D, V)
    section_attributes(D, V)
    section_prevention(D, V)
    section_moderation(D, V)

    if TEMPLATE.exists():
        text = Template(TEMPLATE.read_text()).substitute({k: str(v) for k, v in V.items()})
        OUT.write_text(text)
        print(f"Wrote {OUT.relative_to(ROOT)} and {len(list(FIG.glob('*.png')))} figures in reports/figures/")
    else:
        for k, v in V.items():
            print(f"--- {k}\n{v}" if "\n" in str(v) else f"{k} = {v}")


if __name__ == "__main__":
    main()
