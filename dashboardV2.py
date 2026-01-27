# ---------------------------------------------------------
# Media Intelligence Dashboard (dashboardV2.py)
# Fully merged with defensive fixes
# ---------------------------------------------------------

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# PATH TO MASTER JSON
# ---------------------------------------------------------

MASTER_JSON = Path(r"./master.json")

# ---------------------------------------------------------
# SENTIMENT LABEL CLEANUP
# ---------------------------------------------------------

SENTIMENT_LABEL_DISPLAY = {
    "very_negative": "Very Negative",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "very_positive": "Very Positive",
}

SENTIMENT_ORDER = [
    "Very Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Very Positive",
]

SENTIMENT_WEIGHTS = {
    "Very Positive": 2,
    "Positive": 1,
    "Neutral": 0,
    "Negative": -1,
    "Very Negative": -2,
}

# ---------------------------------------------------------
# TOPIC BUCKET MAPPING (data-driven taxonomy)
# ---------------------------------------------------------

TOPIC_BUCKET_MAP = {
    # Performance & Strategy
    "Financial Performance & Market Position": "Performance & Strategy",
    "Strategy & Transformation": "Performance & Strategy",

    # Customer & Brand Experience
    "Customer Experience & Service Delivery": "Customer & Brand Experience",
    "Corporate Reputation & Public Perception": "Customer & Brand Experience",
    "Products & Offerings": "Customer & Brand Experience",

    # Governance, Leadership & Accountability
    "Leadership & Governance": "Governance, Leadership & Accountability",
    "Regulation & Compliance": "Governance, Leadership & Accountability",

    # Workforce, Culture & Operations
    "Workforce, Culture & Operations": "Workforce, Culture & Operations",
}

BUCKET_SHORT = {
    "Performance & Strategy": "Performance\n& Strategy",
    "Customer & Brand Experience": "Customer\n& Brand",
    "Governance, Leadership & Accountability": "Governance\n& Leadership",
    "Workforce, Culture & Operations": "Workforce\n& Operations",
}

# ---------------------------------------------------------
# HYBRID TOPIC THRESHOLDS (from your setup)
# ---------------------------------------------------------

TOPIC_THRESHOLD = 0.18
TOPIC_ALPHA_EMBED = 0.6   # embeddings
TOPIC_ALPHA_NLI = 0.4     # DeBERTa-MNLI

# ---------------------------------------------------------
# LOAD MASTER JSON
# ---------------------------------------------------------

@st.cache_data
def load_master():
    data = json.loads(MASTER_JSON.read_text(encoding="utf-8"))

    df_sent = pd.DataFrame(data.get("sentences", []))
    df_topics = pd.DataFrame(data.get("topics", []))
    df_entities = pd.DataFrame(data.get("entities_corpus", []))
    df_linked = pd.DataFrame(data.get("entities_linked", []))
    df_ent_sent = pd.DataFrame(data.get("entity_sentiment", []))
    df_ent_time = pd.DataFrame(data.get("entity_timeline", []))

    # Always create sentiment_display
    if "label_5" in df_sent.columns:
        df_sent["sentiment_display"] = df_sent["label_5"].map(SENTIMENT_LABEL_DISPLAY)
    elif "sentiment" in df_sent.columns:
        df_sent["sentiment_display"] = df_sent["sentiment"].map(SENTIMENT_LABEL_DISPLAY)
    elif "sentiment_label" in df_sent.columns:
        df_sent["sentiment_display"] = df_sent["sentiment_label"].map(SENTIMENT_LABEL_DISPLAY)
    elif "sentiment_class" in df_sent.columns:
        df_sent["sentiment_display"] = df_sent["sentiment_class"].map(SENTIMENT_LABEL_DISPLAY)
    elif "sentiment_category" in df_sent.columns:
        df_sent["sentiment_display"] = df_sent["sentiment_category"].map(SENTIMENT_LABEL_DISPLAY)
    else:
        df_sent["sentiment_display"] = "Neutral"

    return data, df_sent, df_topics, df_entities, df_linked, df_ent_sent, df_ent_time

# ---------------------------------------------------------
# APPLY TOPIC BUCKETS
# ---------------------------------------------------------

def apply_topic_buckets(df_sent, df_topics):
    df_sent = df_sent.copy()
    df_topics = df_topics.copy()

    df_sent["topic_bucket"] = df_sent["topic_name"].map(TOPIC_BUCKET_MAP).fillna("None")
    df_topics["topic_bucket"] = df_topics["topic_name"].map(TOPIC_BUCKET_MAP).fillna("None")

    bucket_sizes = (
        df_sent[df_sent["topic_bucket"].ne("None")]
        .groupby("topic_bucket")["sentence"]
        .count()
        .reset_index()
        .rename(columns={"sentence": "size"})
        .sort_values("size", ascending=False)
    )

    return df_sent, df_topics, bucket_sizes

# ---------------------------------------------------------
# SECTION 2 — Bucket Analytics (Polarity, Summaries, Drivers)
# ---------------------------------------------------------

def compute_bucket_polarity(df_sent: pd.DataFrame) -> pd.DataFrame:
    df = df_sent[df_sent["topic_bucket"].ne("None")].copy()
    df = df[df["topic_bucket"].ne("Other")]

    if df.empty:
        return pd.DataFrame()

    df["count"] = 1

    pivot = (
        df.groupby(["topic_bucket", "sentiment_display"])["count"]
        .sum()
        .reset_index()
    )

    totals = (
        pivot.groupby("topic_bucket")["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "total"})
    )

    pivot = pivot.merge(totals, on="topic_bucket")
    pivot["percent"] = (pivot["count"] / pivot["total"]) * 100

    rows = []
    for bucket in pivot["topic_bucket"].unique():
        sub = pivot[pivot["topic_bucket"] == bucket]

        polarity = sum(
            SENTIMENT_WEIGHTS[row["sentiment_display"]] * row["percent"]
            for _, row in sub.iterrows()
        )

        pos = sub[sub["sentiment_display"].isin(["Positive", "Very Positive"])]["percent"].sum()
        neg = sub[sub["sentiment_display"].isin(["Negative", "Very Negative"])]["percent"].sum()
        neu = sub[sub["sentiment_display"].eq("Neutral")]["percent"].sum()

        rows.append({
            "topic_bucket": bucket,
            "bucket_short": BUCKET_SHORT[bucket],
            "polarity": polarity,
            "positive_percent": pos,
            "negative_percent": neg,
            "neutral_percent": neu,
        })

    return pd.DataFrame(rows)


def generate_bucket_summary(df_sent: pd.DataFrame) -> dict:
    summaries = {}

    df = df_sent[df_sent["topic_bucket"].ne("None")].copy()
    df = df[df["topic_bucket"].ne("Other")]

    if df.empty:
        return summaries

    df["count"] = 1

    pivot = (
        df.groupby(["topic_bucket", "sentiment_display"])["count"]
        .sum()
        .reset_index()
    )

    totals = (
        pivot.groupby("topic_bucket")["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "total"})
    )

    pivot = pivot.merge(totals, on="topic_bucket")
    pivot["percent"] = (pivot["count"] / pivot["total"]) * 100

    for bucket in pivot["topic_bucket"].unique():
        sub = pivot[pivot["topic_bucket"] == bucket]

        polarity = sum(
            SENTIMENT_WEIGHTS[row["sentiment_display"]] * row["percent"]
            for _, row in sub.iterrows()
        )

        pos = sub[sub["sentiment_display"].isin(["Positive", "Very Positive"])]["percent"].sum()
        neg = sub[sub["sentiment_display"].isin(["Negative", "Very Negative"])]["percent"].sum()
        neu = sub[sub["sentiment_display"].eq("Neutral")]["percent"].sum()

        if polarity > 0:
            tone = "strong positive sentiment"
            arrow = "↑"
        elif polarity < 0:
            tone = "strong negative sentiment"
            arrow = "↓"
        else:
            tone = "mixed or neutral sentiment"
            arrow = "→"

        summaries[bucket] = (
            f"{arrow} {bucket} shows {tone}. "
            f"Positive coverage: {pos:.1f}%. "
            f"Negative coverage: {neg:.1f}%. "
            f"Neutral coverage: {neu:.1f}%. "
            f"Polarity score: {polarity:.1f}."
        )

    return summaries


def get_sentiment_drivers(df_sent: pd.DataFrame, top_n: int = 5) -> dict:
    df = df_sent[df_sent["topic_bucket"].ne("None")].copy()
    df = df[df["topic_bucket"].ne("Other")]

    df["sentiment_weight"] = df["sentiment_display"].map(SENTIMENT_WEIGHTS)
    df["driver_score"] = df["sentiment_weight"] * df["topic_score"]

    drivers = {}

    for bucket in df["topic_bucket"].unique():
        sub = df[df["topic_bucket"] == bucket]

        pos = (
            sub[sub["sentiment_weight"] > 0]
            .sort_values("driver_score", ascending=False)
            .head(top_n)
        )
        neg = (
            sub[sub["sentiment_weight"] < 0]
            .sort_values("driver_score")
            .head(top_n)
        )

        drivers[bucket] = {
            "positive": pos[["sentence", "sentiment_display", "topic_score", "driver_score"]],
            "negative": neg[["sentence", "sentiment_display", "topic_score", "driver_score"]],
        }

    return drivers


def get_global_sentiment_drivers(df_sent: pd.DataFrame, top_n: int = 3):
    df = df_sent[df_sent["topic_bucket"].ne("None")].copy()

    df["sentiment_weight"] = df["sentiment_display"].map(SENTIMENT_WEIGHTS)
    df["driver_score"] = df["sentiment_weight"] * df["topic_score"]

    pos = (
        df[df["sentiment_weight"] > 0]
        .sort_values("driver_score", ascending=False)
        .head(top_n)
    )
    neg = (
        df[df["sentiment_weight"] < 0]
        .sort_values("driver_score")
        .head(top_n)
    )

    return pos, neg

# ---------------------------------------------------------
# SECTION 3 — Visuals (Heatmaps, Drift, Purity)
# ---------------------------------------------------------

def bucket_sentiment_heatmap(df_sent: pd.DataFrame):
    df = df_sent[df_sent["topic_bucket"].ne("None")].copy()
    df = df[df["topic_bucket"].ne("Other")]

    if df.empty:
        return None

    df["count"] = 1

    pivot = (
        df.groupby(["topic_bucket", "sentiment_display"])["count"]
        .sum()
        .reset_index()
    )

    totals = (
        pivot.groupby("topic_bucket")["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "total"})
    )

    pivot = pivot.merge(totals, on="topic_bucket")
    pivot["percent"] = (pivot["count"] / pivot["total"]) * 100
    pivot["bucket_short"] = pivot["topic_bucket"].map(BUCKET_SHORT)

    chart = (
        alt.Chart(pivot)
        .mark_rect()
        .encode(
            x=alt.X(
                "sentiment_display:N",
                sort=SENTIMENT_ORDER,
                axis=alt.Axis(
                    labelAngle=0,
                    labelLimit=200,
                    labelOverlap=False,
                    title="Sentiment",
                ),
            ),
            y=alt.Y(
                "bucket_short:N",
                axis=alt.Axis(
                    labelAngle=0,
                    labelLimit=200,
                    labelOverlap=False,
                    title="Topic Bucket",
                ),
            ),
            color=alt.Color(
                "percent:Q",
                scale=alt.Scale(scheme="blues"),
                title="Percent of Bucket",
            ),
            tooltip=[
                "topic_bucket",
                "sentiment_display",
                "percent",
                "count",
                "total",
            ],
        )
    )

    return chart


def topic_drift_heatmap(df_sent: pd.DataFrame):
    if "topic_bucket" not in df_sent.columns:
        return None

    df = df_sent.copy()
    df = df[df["topic_bucket"].ne("None")]
    df = df[df["topic_bucket"].ne("Other")]

    if df.empty:
        return None

    df["bucket_short"] = df["topic_bucket"].map(BUCKET_SHORT)
    df["count"] = 1

    pivot = (
        df.groupby(["topic_name", "bucket_short"])["count"]
        .sum()
        .reset_index()
    )

    totals = (
        pivot.groupby("topic_name")["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "total"})
    )

    pivot = pivot.merge(totals, on="topic_name")
    pivot["percent"] = (pivot["count"] / pivot["total"]) * 100

    chart = (
        alt.Chart(pivot)
        .mark_rect()
        .encode(
            x=alt.X(
                "bucket_short:N",
                axis=alt.Axis(
                    labelAngle=0,
                    labelLimit=200,
                    labelOverlap=False,
                    title="Bucket",
                ),
            ),
            y=alt.Y(
                "topic_name:N",
                axis=alt.Axis(
                    labelAngle=0,
                    labelLimit=300,
                    labelOverlap=False,
                    title="Fine Topic",
                ),
            ),
            color=alt.Color(
                "percent:Q",
                scale=alt.Scale(scheme="greens"),
                title="Percent of Topic",
            ),
            tooltip=[
                "topic_name",
                "bucket_short",
                "percent",
                "count",
                "total",
            ],
        )
    )

    return chart


def compute_topic_purity(df_sent: pd.DataFrame) -> pd.DataFrame:
    # Defensive: ensure dataframe and column exist
    if df_sent is None or df_sent.empty:
        return pd.DataFrame()
    if "topic_bucket" not in df_sent.columns:
        return pd.DataFrame()

    df = df_sent.copy()
    df = df[df["topic_bucket"].ne("None")]
    df = df[df["topic_bucket"].ne("Other")]

    if df.empty:
        return pd.DataFrame()

    df["count"] = 1

    pivot = (
        df.groupby(["topic_name", "topic_bucket"])["count"]
        .sum()
        .reset_index()
    )

    totals = (
        pivot.groupby("topic_name")["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "total"})
    )

    pivot = pivot.merge(totals, on="topic_name")
    pivot["percent"] = (pivot["count"] / pivot["total"]) * 100

    purity = (
        pivot.groupby("topic_name")["percent"]
        .max()
        .reset_index()
        .rename(columns={"percent": "purity"})
    )

    return purity.sort_values("purity", ascending=False)


# ---------------------------------------------------------
# SECTION 4 — Executive Summary Page
# ---------------------------------------------------------

def build_executive_summary(df_polarity: pd.DataFrame) -> str:
    if df_polarity.empty:
        return "No sentiment data available for an executive summary."

    lines = []

    for _, row in df_polarity.sort_values("polarity", ascending=False).iterrows():
        bucket = row["topic_bucket"]
        pol = row["polarity"]
        pos = row["positive_percent"]
        neg = row["negative_percent"]
        neu = row["neutral_percent"]

        if pol > 0:
            direction = "is a positive area with favourable coverage."
        elif pol < 0:
            direction = "is a negative area with critical coverage."
        else:
            direction = "shows a mixed or neutral sentiment profile."

        lines.append(
            f"{bucket} {direction} "
            f"Approx. {pos:.0f}% positive, {neg:.0f}% negative, and {neu:.0f}% neutral."
        )

    return " ".join(lines)


def render_executive_summary_page(df_sent: pd.DataFrame):
    st.header("Executive Summary")

    df_polarity = compute_bucket_polarity(df_sent)
    summary_text = build_executive_summary(df_polarity)

    st.markdown("### Narrative Overview")
    st.write(summary_text)

    st.markdown("### Polarity by Bucket")

    if not df_polarity.empty:
        chart_pol = (
            alt.Chart(df_polarity)
            .mark_bar()
            .encode(
                x=alt.X(
                    "bucket_short:N",
                    sort="-y",
                    axis=alt.Axis(
                        labelAngle=0,
                        labelLimit=200,
                        labelOverlap=False,
                        title="Bucket",
                    ),
                ),
                y=alt.Y("polarity:Q", title="Polarity Score"),
                color=alt.condition(
                    alt.datum.polarity > 0,
                    alt.value("#2ca02c"),
                    alt.value("#d62728"),
                ),
                tooltip=[
                    "topic_bucket",
                    "polarity",
                    "positive_percent",
                    "negative_percent",
                    "neutral_percent",
                ],
            )
        )
        st.altair_chart(chart_pol, use_container_width=True)
    else:
        st.write("No polarity data available.")

    st.markdown("### Top Sentiment Drivers (Global)")

    pos, neg = get_global_sentiment_drivers(df_sent, top_n=3)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Positive Drivers**")
        if not pos.empty:
            st.dataframe(
                pos[
                    [
                        "sentence",
                        "topic_name",
                        "topic_bucket",
                        "sentiment_display",
                        "driver_score",
                    ]
                ]
            )
        else:
            st.write("No positive drivers available.")

    with col2:
        st.markdown("**Top Negative Drivers**")
        if not neg.empty:
            st.dataframe(
                neg[
                    [
                        "sentence",
                        "topic_name",
                        "topic_bucket",
                        "sentiment_display",
                        "driver_score",
                    ]
                ]
            )
        else:
            st.write("No negative drivers available.")

# ---------------------------------------------------------
# SECTION 5 — Topic Buckets Page
# ---------------------------------------------------------

def render_topic_buckets_page(df_sent: pd.DataFrame, df_topics: pd.DataFrame, bucket_sizes: pd.DataFrame):
    st.header("High‑Level Topic Buckets")

    # Defensive: no data, no page
    if df_sent is None or df_sent.empty:
        st.write("No sentence data available.")
        return

    # Ensure topic_bucket exists
    if "topic_bucket" not in df_sent.columns:
        df_sent, df_topics, bucket_sizes = apply_topic_buckets(df_sent, df_topics)

    df_sent_b = df_sent[df_sent["topic_bucket"].ne("None")].copy()
    df_sent_b = df_sent_b[df_sent_b["topic_bucket"].ne("Other")]


    st.markdown("### Bucket Sizes")
    if bucket_sizes is not None and not bucket_sizes.empty:
        st.dataframe(bucket_sizes[bucket_sizes["topic_bucket"].ne("None")])
    else:
        st.write("No bucket size data available.")

    df_polarity = compute_bucket_polarity(df_sent_b)

    st.markdown("### Bucket Polarity Ranking")
    if not df_polarity.empty:
        st.dataframe(df_polarity.sort_values("polarity", ascending=False))
    else:
        st.write("No polarity data available.")

    st.markdown("### Polarity by Bucket")

    if not df_polarity.empty:
        chart_pol = (
            alt.Chart(df_polarity)
            .mark_bar()
            .encode(
                x=alt.X(
                    "bucket_short:N",
                    sort="-y",
                    axis=alt.Axis(
                        labelAngle=0,
                        labelLimit=200,
                        labelOverlap=False,
                        title="Bucket",
                    ),
                ),
                y=alt.Y("polarity:Q", title="Polarity Score"),
                color=alt.condition(
                    alt.datum.polarity > 0,
                    alt.value("#2ca02c"),
                    alt.value("#d62728"),
                ),
                tooltip=[
                    "topic_bucket",
                    "polarity",
                    "positive_percent",
                    "negative_percent",
                    "neutral_percent",
                ],
            )
        )
        st.altair_chart(chart_pol, use_container_width=True)
    else:
        st.write("No polarity scores to display.")

    st.markdown("### Bucket × Sentiment Heatmap")

    heatmap = bucket_sentiment_heatmap(df_sent_b)
    if heatmap is not None:
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.write("No data available for bucket heatmap.")

    st.markdown("### Topic Drift (Fine Topics → Buckets)")

    drift_chart = topic_drift_heatmap(df_sent_b)
    if drift_chart is not None:
        st.altair_chart(drift_chart, use_container_width=True)
    else:
        st.write("No data available for topic drift.")

    st.markdown("### Topic Purity Scores")

    purity = compute_topic_purity(df_sent_b)
    if not purity.empty:
        st.dataframe(purity)
    else:
        st.write("No purity data available.")

    st.markdown("### Top Sentiment Drivers (by Bucket)")

    drivers = get_sentiment_drivers(df_sent_b)
    for bucket, d in drivers.items():
        st.markdown(f"#### {bucket}")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Positive Drivers**")
            if not d["positive"].empty:
                st.dataframe(d["positive"])
            else:
                st.write("No positive drivers.")

        with col2:
            st.markdown("**Top Negative Drivers**")
            if not d["negative"].empty:
                st.dataframe(d["negative"])
            else:
                st.write("No negative drivers.")

    st.markdown("### Narrative Summaries")

    summaries = generate_bucket_summary(df_sent_b)
    if summaries:
        for bucket, text in summaries.items():
            st.markdown(f"**{bucket}**")
            st.write(text)
    else:
        st.write("No summaries available.")

# ---------------------------------------------------------
# SECTION 6 — Overview, Topic Explorer, Sentence Inspector
# ---------------------------------------------------------

def render_overview_page(df_sent, df_topics, df_articles):
    st.header("Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Sentences", len(df_sent))

    with col2:
        st.metric("Unique Topics", df_topics["topic_name"].nunique() if not df_topics.empty else 0)

    with col3:
        st.metric("Total Articles", len(df_articles) if df_articles is not None else 0)

    st.markdown("### Sample of Sentences")
    st.dataframe(df_sent.head(20))


def render_topic_explorer_page(df_sent):
    st.header("Topic Explorer")

    topic_list = sorted(df_sent["topic_name"].dropna().unique())
    topic = st.selectbox("Select a topic", topic_list)

    df_t = df_sent[df_sent["topic_name"] == topic].copy()

    st.markdown(f"### Sentences for: **{topic}**")
    st.dataframe(
        df_t[
            [
                "sentence",
                "topic_name",
                "topic_bucket",
                "sentiment_display",
                "topic_score",
            ]
        ]
    )


def render_sentence_inspector_page(df_sent):
    st.header("Sentence Inspector")

    query = st.text_input("Search text in sentences")

    if query:
        df_q = df_sent[
            df_sent["sentence"].str.contains(query, case=False, na=False)
        ].copy()

        st.write(f"Found **{len(df_q)}** matching sentences.")
        st.dataframe(
            df_q[
                [
                    "sentence",
                    "topic_name",
                    "topic_bucket",
                    "sentiment_display",
                    "topic_score",
                ]
            ]
        )
    else:
        st.write("Enter text above to search through all sentences.")

# ---------------------------------------------------------
# SECTION 7 — Entity Explorer, Article Browser, Search, main()
# ---------------------------------------------------------

def render_entity_explorer_page(df_entities, df_sent):
    st.header("Entity Explorer")

    if df_entities.empty:
        st.write("No entity data available.")
        return

    entity_list = sorted(df_entities["entity"].dropna().unique())
    entity = st.selectbox("Select an entity", entity_list)

    df_e = df_entities[df_entities["entity"] == entity].copy()
    st.markdown(f"### Mentions of: **{entity}**")
    st.dataframe(df_e)

    if "sentence_id" in df_e.columns and "sentence_id" in df_sent.columns:
        linked = df_sent[df_sent["sentence_id"].isin(df_e["sentence_id"])]
        if not linked.empty:
            st.markdown("### Sentences mentioning this entity")
            st.dataframe(
                linked[
                    [
                        "sentence",
                        "topic_name",
                        "topic_bucket",
                        "sentiment_display",
                        "topic_score",
                    ]
                ]
            )


def render_article_browser_page(df_articles, df_sent):
    st.header("Article Browser")

    if df_articles.empty:
        st.write("No article data available.")
        return

    article_ids = df_articles["article_id"].tolist()
    article_id = st.selectbox("Select an article ID", article_ids)

    df_a = df_articles[df_articles["article_id"] == article_id].copy()
    st.markdown("### Article Metadata")
    st.dataframe(df_a)

    df_s = df_sent[df_sent["article_id"] == article_id].copy()
    st.markdown("### Sentences in this Article")
    st.dataframe(
        df_s[
            [
                "sentence",
                "topic_name",
                "topic_bucket",
                "sentiment_display",
                "topic_score",
            ]
        ]
    )


def render_search_page(df_sent, df_articles):
    st.header("Search")

    query = st.text_input("Search across sentences and articles")

    if not query:
        st.write("Enter a search term to begin.")
        return

    df_s = df_sent[df_sent["sentence"].str.contains(query, case=False, na=False)]
    st.markdown("### Sentence Matches")
    st.write(f"{len(df_s)} matches")
    st.dataframe(
        df_s[
            [
                "sentence",
                "topic_name",
                "topic_bucket",
                "sentiment_display",
                "topic_score",
            ]
        ]
    )

    if not df_articles.empty and "title" in df_articles.columns:
        df_a = df_articles[df_articles["title"].str.contains(query, case=False, na=False)]
        st.markdown("### Article Title Matches")
        st.write(f"{len(df_a)} matches")
        st.dataframe(df_a)

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Media Intelligence Dashboard",
        layout="wide",
    )

    st.title("Media Intelligence Dashboard")

    (
        data,
        df_sent,
        df_topics,
        df_entities,
        df_linked,
        df_ent_sent,
        df_ent_time,
    ) = load_master()

    df_sent, df_topics, bucket_sizes = apply_topic_buckets(df_sent, df_topics)
    df_articles = pd.DataFrame(data.get("articles", []))

    # Safety: ensure topic_bucket exists before routing
    if "topic_bucket" not in df_sent.columns:
        df_sent, df_topics, bucket_sizes = apply_topic_buckets(df_sent, df_topics)

    page = st.sidebar.selectbox(
        "Page",
        [
            "Executive Summary",
            "Topic Buckets",
            "Overview",
            "Topic Explorer",
            "Sentence Inspector",
            "Entity Explorer",
            "Article Browser",
            "Search",
        ],
    )

    if page == "Executive Summary":
        render_executive_summary_page(df_sent)
    elif page == "Topic Buckets":
        render_topic_buckets_page(df_sent, df_topics, bucket_sizes)
    elif page == "Overview":
        render_overview_page(df_sent, df_topics, df_articles)
    elif page == "Topic Explorer":
        render_topic_explorer_page(df_sent)
    elif page == "Sentence Inspector":
        render_sentence_inspector_page(df_sent)
    elif page == "Entity Explorer":
        render_entity_explorer_page(df_entities, df_sent)
    elif page == "Article Browser":
        render_article_browser_page(df_articles, df_sent)
    elif page == "Search":
        render_search_page(df_sent, df_articles)


if __name__ == "__main__":
    main()




