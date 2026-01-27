import json
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

MASTER_JSON = Path(r"C:\Users\henry\OneDrive\Desktop\Presschoice\Articles\master.json")

# ---------------------------------------------------------
# SENTIMENT LABEL CLEANUP
# ---------------------------------------------------------

SENTIMENT_LABEL_DISPLAY = {
    "very_negative": "Very Negative",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "very_positive": "Very Positive"
}

SENTIMENT_ORDER = [
    "Very Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Very Positive"
]

# ---------------------------------------------------------
# TOPIC BUCKET MAPPING
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
    "Performance & Strategy": "Performance",
    "Customer & Brand Experience": "Customer",
    "Governance, Leadership & Accountability": "Governance",
    "Workforce, Culture & Operations": "Workforce",
}



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

    # Add clean sentiment labels
    if "label_5" in df_sent.columns:
        df_sent["sentiment_display"] = df_sent["label_5"].map(SENTIMENT_LABEL_DISPLAY)

    return data, df_sent, df_topics, df_entities, df_linked, df_ent_sent, df_ent_time

# ---------------------------------------------------------
# BUCKET APPLICATION & ANALYTICS
# ---------------------------------------------------------

def apply_topic_buckets(df_sent, df_topics):
    df_sent = df_sent.copy()
    df_topics = df_topics.copy()

    df_sent["topic_bucket"] = df_sent["topic_name"].map(TOPIC_BUCKET_MAP).fillna("Other")
    df_topics["topic_bucket"] = df_topics["topic_name"].map(TOPIC_BUCKET_MAP).fillna("Other")

    bucket_sizes = (
        df_sent[df_sent["topic_bucket"] != "None"]
        .groupby("topic_bucket")["sentence"]
        .count()
        .reset_index()
        .rename(columns={"sentence": "size"})
    )

    return df_sent, df_topics, bucket_sizes

def bucket_sentiment_heatmap(df_sent):
    df = df_sent[df_sent["topic_bucket"] != "None"].copy()
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

    chart = (
        alt.Chart(pivot)
        .mark_rect()
        .encode(
            x=alt.X("sentiment_display:N", sort=SENTIMENT_ORDER, axis=alt.Axis(labelAngle=45)),
            y=alt.Y("topic_bucket:N", axis=alt.Axis(labelAngle=0)),
            color=alt.Color("percent:Q", scale=alt.Scale(scheme="redyellowgreen"), title="Percent of bucket"),
            tooltip=["topic_bucket", "sentiment_display", "percent", "count", "total"]
        )
    )

    return chart

def generate_bucket_summary(df_sent):
    summaries = {}

    df = df_sent[df_sent["topic_bucket"] != "None"].copy()
    df = df[df["topic_bucket"] != "Other"]
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

    weights = {
        "Very Positive": 2,
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1,
        "Very Negative": -2
    }

    for bucket in pivot["topic_bucket"].unique():
        sub = pivot[pivot["topic_bucket"] == bucket]

        polarity = sum(
            weights[row["sentiment_display"]] * row["percent"]
            for _, row in sub.iterrows()
        )

        pos = sub[sub["sentiment_display"].isin(["Positive", "Very Positive"])]["percent"].sum()
        neg = sub[sub["sentiment_display"].isin(["Negative", "Very Negative"])]["percent"].sum()
        neu = sub[sub["sentiment_display"] == "Neutral"]["percent"].sum()

        if polarity > 0:
            tone = "strong positive sentiment"
            arrow = "↑"
        elif polarity < 0:
            tone = "strong negative sentiment"
            arrow = "↓"
        else:
            tone = "predominantly neutral sentiment"
            arrow = "→"

        summaries[bucket] = (
            f"{arrow} {bucket} shows {tone}. "
            f"Positive coverage: {pos:.1f}%. "
            f"Negative coverage: {neg:.1f}%. "
            f"Neutral coverage: {neu:.1f}%. "
            f"Polarity score: {polarity:.1f}."
        )

    return summaries


def compute_bucket_polarity(df_sent):
    df = df_sent[df_sent["topic_bucket"] != "None"].copy()
    df = df[df["topic_bucket"] != "Other"]
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

    weights = {
        "Very Positive": 2,
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1,
        "Very Negative": -2
    }

    rows = []
    for bucket in pivot["topic_bucket"].unique():
        sub = pivot[pivot["topic_bucket"] == bucket]

        polarity = sum(
            weights[row["sentiment_display"]] * row["percent"]
            for _, row in sub.iterrows()
        )

        pos = sub[sub["sentiment_display"].isin(["Positive", "Very Positive"])]["percent"].sum()
        neg = sub[sub["sentiment_display"].isin(["Negative", "Very Negative"])]["percent"].sum()
        neu = sub[sub["sentiment_display"] == "Neutral"]["percent"].sum()

        rows.append({
            "topic_bucket": bucket,
            "bucket_short": BUCKET_SHORT[bucket],
            "polarity": polarity,
            "positive_percent": pos,
            "negative_percent": neg,
            "neutral_percent": neu
        })

    return pd.DataFrame(rows)

# --------------------------------------------------------- Setiment drivers

def get_sentiment_drivers(df_sent, top_n=5):
    df = df_sent[df_sent["topic_bucket"] != "None"].copy()
    df = df[df["topic_bucket"] != "Other"]

    weights = {
        "Very Positive": 2,
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1,
        "Very Negative": -2
    }

    df["sentiment_weight"] = df["sentiment_display"].map(weights)
    df["driver_score"] = df["sentiment_weight"] * df["topic_score"]

    drivers = {}

    for bucket in df["topic_bucket"].unique():
        sub = df[df["topic_bucket"] == bucket]

        pos = sub[sub["sentiment_weight"] > 0].sort_values("driver_score", ascending=False).head(top_n)
        neg = sub[sub["sentiment_weight"] < 0].sort_values("driver_score").head(top_n)

        drivers[bucket] = {
            "positive": pos[["sentence", "sentiment_display", "topic_score", "driver_score"]],
            "negative": neg[["sentence", "sentiment_display", "topic_score", "driver_score"]]
        }

    return drivers

# topic drift
def topic_drift_heatmap(df_sent):
    df = df_sent.copy()
    df = df[df["topic_bucket"] != "Other"]
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
            x=alt.X("bucket_short:N", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("topic_name:N", axis=alt.Axis(labelAngle=0)),
            color=alt.Color("percent:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["topic_name", "bucket_short", "percent"]
        )
    )

    return chart

#topic purity

def compute_topic_purity(df_sent):
    df = df_sent.copy()
    df = df[df["topic_bucket"] != "Other"]

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

    return purity

#topic overlap

def compute_topic_overlap(df_sent):
    df = df_sent.copy()
    df = df[df["topic_bucket"] != "Other"]

    df["count"] = 1

    pivot = (
        df.groupby(["topic_bucket", "topic_name"])["count"]
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

    return pivot


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------

master, df_sent, df_topics, df_entities, df_linked, df_ent_sent, df_ent_time = load_master()

st.set_page_config(
    page_title="Hybrid Topic & Sentiment Dashboard",
    layout="wide"
)

st.title("📊 Hybrid Topic & Sentiment Intelligence Dashboard")
st.caption("Powered by mpnet embeddings + DeBERTa-MNLI + entity/context nudging")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Overview",
        "Topic Buckets",
        "Topic Explorer",
        "Sentence Inspector",
        "Entity Explorer",
        "Article Browser",
        "Search"
    ]
)

# ---------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------

if page == "Overview":
    st.subheader("Sentiment Distribution (5-class)")

    if "sentiment_5" in master:
        df = pd.DataFrame([
            {"sentiment": SENTIMENT_LABEL_DISPLAY[k], "value": v}
            for k, v in master["sentiment_5"].items()
        ])

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("sentiment:N", sort=SENTIMENT_ORDER, axis=alt.Axis(labelAngle=45)),
                y="value:Q",
                color="sentiment:N"
            )
        )
        st.altair_chart(chart, use_container_width=True)

    # Show % None
    none_pct = (df_sent["topic_name"] == "None").mean() * 100
    st.markdown(f"**Sentences with no strong topic assignment (‘None’): {none_pct:.1f}%**")

    st.subheader("Topic Sizes")
    df_topics_clean = df_topics[df_topics["topic_name"] != "None"]

    if not df_topics_clean.empty:
        df_topics_sorted = df_topics_clean.sort_values("size", ascending=False)
        chart = (
            alt.Chart(df_topics_sorted)
            .mark_bar()
            .encode(
                x=alt.X("topic_name:N", sort="-y", axis=alt.Axis(labelAngle=45)),
                y="size:Q",
                color="topic_name:N"
            )
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Topic × Sentiment Heatmap (percent per topic)")

    df_heat = df_sent[df_sent["topic_name"] != "None"].copy()

    if not df_heat.empty:
        df_heat["count"] = 1

        pivot = (
            df_heat
            .groupby(["topic_name", "sentiment_display"])["count"]
            .sum()
            .reset_index()
        )

        totals = (
            pivot
            .groupby("topic_name")["count"]
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
                x=alt.X("sentiment_display:N", sort=SENTIMENT_ORDER, axis=alt.Axis(labelAngle=45)),
                y=alt.Y("topic_name:N", axis=alt.Axis(labelAngle=0)),
                color=alt.Color("percent:Q", scale=alt.Scale(scheme="redyellowgreen")),
                tooltip=["topic_name", "sentiment_display", "percent", "count", "total"]
            )
        )

        st.altair_chart(chart, use_container_width=True)

    st.subheader("Weighted Topic Strength (sum of topic scores)")

    df_strength = (
        df_sent[df_sent["topic_name"] != "None"]
        .groupby("topic_name")["topic_score"]
        .sum()
        .reset_index()
    )

    if not df_strength.empty:
        chart = (
            alt.Chart(df_strength)
            .mark_bar()
            .encode(
                x=alt.X("topic_name:N", sort="-y", axis=alt.Axis(labelAngle=45)),
                y=alt.Y("topic_score:Q", title="Total Topic Strength"),
                color="topic_name:N"
            )
        )

        st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------
# PAGE: TOPIC BUCKETS
# ---------------------------------------------------------

elif page == "Topic Buckets":
    st.subheader("High-Level Topic Buckets")

    df_sent_b, df_topics_b, bucket_sizes = apply_topic_buckets(df_sent, df_topics)
    df_sent_b = df_sent_b[df_sent_b["topic_bucket"] != "Other"]

    # Bucket sizes
    st.markdown("### Bucket Sizes")
    st.dataframe(bucket_sizes[bucket_sizes["topic_bucket"] != "Other"])

    # Polarity ranking
    st.markdown("### Bucket Polarity Ranking")
    df_polarity = compute_bucket_polarity(df_sent_b)
    st.dataframe(df_polarity.sort_values("polarity", ascending=False))

    # Polarity bar chart
    st.markdown("### Polarity by Bucket")
    chart_pol = (
        alt.Chart(df_polarity)
        .mark_bar()
        .encode(
            x=alt.X("bucket_short:N", sort="-y"),
            y=alt.Y("polarity:Q"),
            color=alt.condition(
                alt.datum.polarity > 0,
                alt.value("#2ca02c"),
                alt.value("#d62728")
            ),
            tooltip=["topic_bucket", "polarity", "positive_percent", "negative_percent", "neutral_percent"]
        )
    )
    st.altair_chart(chart_pol, use_container_width=True)

    # Heatmap
    st.markdown("### Bucket × Sentiment Heatmap")
    chart = bucket_sentiment_heatmap(df_sent_b)
    st.altair_chart(chart, use_container_width=True)

    # Topic Drift
    st.markdown("### Topic Drift (Fine Topics → Buckets)")
    drift_chart = topic_drift_heatmap(df_sent_b)
    st.altair_chart(drift_chart, use_container_width=True)

    # Topic Purity
    st.markdown("### Topic Purity Scores")
    purity = compute_topic_purity(df_sent_b)
    st.dataframe(purity.sort_values("purity", ascending=False))

    # Top Drivers
    st.markdown("### Top Sentiment Drivers")
    drivers = get_sentiment_drivers(df_sent_b)

    for bucket, d in drivers.items():
        st.markdown(f"#### {bucket}")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Positive Drivers**")
            st.dataframe(d["positive"])

        with col2:
            st.markdown("**Top Negative Drivers**")
            st.dataframe(d["negative"])

    # Executive Summary
    st.markdown("### Executive Summary")
    summaries = generate_bucket_summary(df_sent_b)
    for bucket, text in summaries.items():
        st.markdown(f"**{bucket}**")
        st.write(text)


# ---------------------------------------------------------
# PAGE: TOPIC EXPLORER
# ---------------------------------------------------------

elif page == "Topic Explorer":
    st.subheader("Topic Explorer")

    df_topics["definition"] = df_topics["definition"].astype(str)

    topic_choice = st.selectbox("Select a topic", df_topics["topic_name"].tolist())

    topic_info = df_topics[df_topics["topic_name"] == topic_choice].iloc[0]
    st.markdown(f"### {topic_choice}")
    st.write(topic_info["definition"])

    df_topic_sent = df_sent[df_sent["topic_name"] == topic_choice]

    st.markdown("#### Representative Sentences")
    for s in df_topic_sent.sort_values("topic_score", ascending=False).head(5)["sentence"]:
        st.write(f"- {s}")

    st.markdown("#### Topic Score Distribution")
    if not df_topic_sent.empty:
        chart = (
            alt.Chart(df_topic_sent)
            .mark_bar()
            .encode(
                x=alt.X("topic_score:Q", bin=alt.Bin(maxbins=20)),
                y="count()",
                tooltip=["count()"]
            )
        )
        st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------
# PAGE: SENTENCE INSPECTOR
# ---------------------------------------------------------

elif page == "Sentence Inspector":
    st.subheader("Sentence Inspector")

    idx = st.number_input(
        "Sentence index",
        min_value=0,
        max_value=len(df_sent) - 1,
        value=0,
        step=1
    )

    row = df_sent.iloc[idx]

    st.markdown(f"### Sentence {idx}")
    st.write(row["sentence"])

    st.markdown("#### Topic Assignment")
    st.write(f"**Topic:** {row['topic_name']} (score {row['topic_score']:.3f})")

    st.markdown("#### Hybrid vs Embedding vs NLI Scores")
    df_scores = pd.DataFrame({
        "topic": list(df_topics["topic_name"][:-1]),
        "embedding": row["topic_scores_embedding"],
        "nli": row["topic_scores_nli"],
        "hybrid": row["topic_scores_hybrid"]
    })

    st.dataframe(df_scores)

    st.markdown("#### Context Features")
    st.json(row.get("context_features", {}))

    st.markdown("#### Entities")
    ents = row.get("entities", [])
    if ents:
        st.dataframe(pd.DataFrame(ents))
    else:
        st.write("No entities detected.")

# ---------------------------------------------------------
# PAGE: ENTITY EXPLORER
# ---------------------------------------------------------

elif page == "Entity Explorer":
    st.subheader("Entity Explorer")

    tab1, tab2, tab3, tab4 = st.tabs(["Corpus Entities", "Linked Entities", "Entity Sentiment", "Timeline"])

    with tab1:
        st.dataframe(df_entities)

    with tab2:
        st.dataframe(df_linked)

    with tab3:
        st.dataframe(df_ent_sent)

    with tab4:
        if not df_ent_time.empty:
            chart = (
                alt.Chart(df_ent_time)
                .mark_line()
                .encode(
                    x="article_index:Q",
                    y="count:Q",
                    color="text:N",
                    tooltip=["text", "article_index", "count"]
                )
            )
            st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------
# PAGE: ARTICLE BROWSER
# ---------------------------------------------------------

elif page == "Article Browser":
    st.subheader("Article Browser")

    articles = master.get("articles", [])
    article_ids = [a["article_id"] for a in articles]

    choice = st.selectbox("Select article", article_ids)

    article = next(a for a in articles if a["article_id"] == choice)

    st.markdown(f"### {article['article_filename']}")
    st.write(article["body"])

    df_article_sent = df_sent[df_sent["article_id"] == choice]

    st.markdown("#### Sentences")
    st.dataframe(df_article_sent[["sentence_index_article", "sentence", "topic_name", "sentiment_display"]])

# ---------------------------------------------------------
# PAGE: SEARCH
# ---------------------------------------------------------

elif page == "Search":
    st.subheader("Search")

    q = st.text_input("Search sentences, entities, or topics")

    if q:
        q_lower = q.lower()

        df_s = df_sent[df_sent["sentence"].str.lower().str.contains(q_lower)]
        df_e = df_entities[df_entities["text"].str.lower().str.contains(q_lower)]
        df_t = df_topics[df_topics["topic_name"].str.lower().str.contains(q_lower)]

        st.markdown("### Sentences")
        st.dataframe(df_s)

        st.markdown("### Entities")
        st.dataframe(df_e)

        st.markdown("### Topics")
        st.dataframe(df_t)
