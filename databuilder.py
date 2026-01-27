import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import re

import nltk
from docx import Document
from nltk.tokenize import sent_tokenize

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# CONFIG
# -------------------------

RAW_DIR = Path(r"C:\Users\henry\OneDrive\Desktop\Presschoice\Articles\Aviva Articles")
MASTER_JSON = Path(r"C:\Users\henry\OneDrive\Desktop\Presschoice\Articles\master.json")
FIGURES_DIR = Path(r"C:\Users\henry\OneDrive\Desktop\Presschoice\Articles\figures")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Sentiment models (3-model ensemble)
MODEL_FINBERT = "ProsusAI/finbert"
MODEL_CARDIFF = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_DISTILFIN = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

# Topic models
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_TOPIC_NLI = "MoritzLaurer/deberta-v3-base-mnli"

# NER model (generic, stable)
MODEL_GENERIC_NER = "dslim/bert-base-NER"

SENTIMENT_LABELS_5 = [
    "very_negative",
    "negative",
    "neutral",
    "positive",
    "very_positive",
]

SENTIMENT_MODELS = ["finbert", "cardiff", "distilfin"]

# -----------------------------------------
# TOPIC DEFINITIONS: MULTI-ANCHOR, JOURNALISM-STYLE
# -----------------------------------------
# 8 topics: the original 5 + Regulation, Strategy, Workforce

TOPIC_DEFINITIONS = {
    "Corporate Reputation & Public Perception": [
        "This sentence comments on how the company is perceived by the public, investors, or the media.",
        "This sentence discusses the company's reputation, image, or standing in the eyes of key stakeholders.",
        "This sentence reflects on whether the company is viewed positively or negatively in broader public debate.",
        "This sentence describes how journalists, commentators, or analysts characterise the company’s behaviour or culture.",
        "This sentence explores trust, credibility, controversy, or praise surrounding the company in the public arena."
    ],

    "Leadership": [
        "This sentence focuses on the actions, decisions, or statements of the company’s senior leadership or board.",
        "This sentence reports on appointments, resignations, succession plans, or reshuffles in the executive team.",
        "This sentence highlights the role, influence, or performance of the CEO, chair, or other key executives.",
        "This sentence raises questions about management judgement, strategic direction, or the quality of leadership."
    ],

    "Customer Experience & Service Delivery": [
        "This sentence reports on how customers experience the company’s services, from day‑to‑day interactions to major complaints.",
        "This sentence highlights issues such as claims handling, response times, or the ease of using the company’s products.",
        "This sentence describes praise or criticism from customers, consumer groups, or ombudsman rulings.",
        "This sentence focuses on service quality, support channels, or changes that affect the customer journey.",
        "This sentence covers efforts to improve customer outcomes, reduce friction, or address service failures."
    ],

    "Products & Offerings": [
        "This sentence explains or introduces a product, service, or solution offered by the company.",
        "This sentence reports on new launches, product changes, or the evolution of the company’s range of offerings.",
        "This sentence discusses how particular products are performing, being positioned, or differentiated in the market.",
        "This sentence focuses on features, coverage, pricing, or design of specific products or services.",
        "This sentence explores how the company’s offerings compare with rivals, or how they meet customer needs."
    ],

    "Financial Performance & Market Position": [
        "This sentence reports on the company’s financial results, including profits, losses, revenue, or earnings guidance.",
        "This sentence describes how the market values the company, referring to valuation, share price moves, or investor reaction.",
        "This sentence focuses on capital strength, solvency, balance sheet health, or key financial ratios.",
        "This sentence examines the company’s competitive position, market share, or performance relative to peers.",
        "This sentence links financial metrics to the company’s outlook, strategy, or ability to deliver returns."
    ],

    "Regulation & Compliance": [
        "This sentence discusses regulatory requirements, oversight, or supervision by bodies such as the FCA or PRA.",
        "This sentence reports on compliance issues, investigations, enforcement actions, or regulatory sanctions affecting the company.",
        "This sentence describes changes in rules, standards, or legislation that impact how the company operates.",
        "This sentence highlights tensions between the company and regulators, or concerns raised by watchdogs and authorities.",
        "This sentence considers how the company is responding to regulatory pressure, scrutiny, or compliance risks.",
        "This sentence examines governance structures, oversight, accountability, and board-level practices that intersect with regulatory responsibilities."
    ],

    "Strategy & Transformation": [
        "This sentence explores the company’s long‑term strategy, transformation plans, or major strategic initiatives.",
        "This sentence reports on restructuring, portfolio changes, disposals, or acquisitions that reshape the business.",
        "This sentence discusses digital transformation, innovation programmes, or shifts in the company’s operating model.",
        "This sentence examines how management is repositioning the business in response to market trends or competitive pressures.",
        "This sentence considers whether the company’s strategy is credible, ambitious, or risky in the eyes of stakeholders."
    ],

    "Workforce, Culture & Operations": [
        "This sentence reports on staffing levels, hiring, redundancies, or changes to the workforce.",
        "This sentence discusses workplace culture, employee morale, diversity, inclusion, or internal tensions.",
        "This sentence highlights industrial relations, union activity, strikes, or disputes involving employees.",
        "This sentence covers operational performance, internal processes, or how effectively the organisation is run day to day.",
        "This sentence examines how the company treats its employees, including pay, conditions, and career prospects."
    ]
}


# topic threshold on final hybrid score (0–1)
TOPIC_THRESHOLD = 0.2
# blending weight between embeddings and NLI
TOPIC_ALPHA_EMBED = 0.7   # embeddings
TOPIC_ALPHA_NLI = 0.3     # DeBERTa-MNLI



# -------------------------
# UTILITIES
# -------------------------

def clean_text(text: str) -> str:
    return (
        text.replace("\r", " ")
            .replace("\n", " ")
            .replace("\t", " ")
            .strip()
    )

def load_docx(path: Path) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

# -------------------------
# STEP 1: BUILD ARTICLES
# -------------------------

def build_articles():
    print("=== BUILDING ARTICLES ===")
    articles = []
    for file in sorted(RAW_DIR.glob("*")):
        suffix = file.suffix.lower()
        if suffix == ".docx":
            text = load_docx(file)
        elif suffix == ".txt":
            text = load_txt(file)
        else:
            continue

        body = clean_text(text)
        if not body:
            continue

        article_id = file.stem
        article_filename = file.name

        articles.append({
            "article_id": article_id,
            "article_filename": article_filename,
            "body": body,
        })

        print(f"Loaded article: {article_filename}")

    for idx, a in enumerate(articles):
        a["article_index"] = idx

    return articles

def create_master_json():
    articles = build_articles()
    master_text = "\n\n".join(a["body"] for a in articles)

    master = {
        "id": "master_corpus",
        "source": "local_combined",
        "created_at": datetime.utcnow().isoformat(),
        "body": master_text,
        "articles": articles,
    }

    MASTER_JSON.write_text(json.dumps(master, indent=2), encoding="utf-8")
    print("Master JSON created:", MASTER_JSON)

# -------------------------
# SENTIMENT HELPERS
# -------------------------

def normalize_probs(probs: list) -> list:
    s = float(sum(probs))
    if s <= 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / s for p in probs]

def majority_label_from_probs(probs: list) -> int:
    return int(max(range(len(probs)), key=lambda i: probs[i]))

def map_3class_to_5class(label_3: str, score: float) -> list:
    label_3 = label_3.lower()
    score = max(0.0, min(1.0, float(score)))

    if label_3.startswith("pos"):
        if score >= 0.7:
            return normalize_probs([0.0, 0.0, 0.05, 0.25, 0.70])
        elif score >= 0.4:
            return normalize_probs([0.0, 0.0, 0.10, 0.70, 0.20])
        else:
            return normalize_probs([0.0, 0.05, 0.40, 0.45, 0.10])

    elif label_3.startswith("neg"):
        if score >= 0.7:
            return normalize_probs([0.70, 0.25, 0.05, 0.0, 0.0])
        elif score >= 0.4:
            return normalize_probs([0.20, 0.70, 0.10, 0.0, 0.0])
        else:
            return normalize_probs([0.10, 0.45, 0.40, 0.05, 0.0])

    else:
        return normalize_probs([0.05, 0.15, 0.60, 0.15, 0.05])

# -------------------------
# MODEL LOADERS
# -------------------------

def load_sentiment_pipelines():
    print("=== LOADING SENTIMENT MODELS ===")
    finbert = hf_pipeline("text-classification", model=MODEL_FINBERT, return_all_scores=True)
    cardiff = hf_pipeline("text-classification", model=MODEL_CARDIFF, return_all_scores=True)
    distilfin = hf_pipeline("text-classification", model=MODEL_DISTILFIN, return_all_scores=True)
    return {
        "finbert": finbert,
        "cardiff": cardiff,
        "distilfin": distilfin,
    }

def load_topic_nli_pipeline():
    print("=== LOADING TOPIC NLI MODEL (DeBERTa-v3-base-mnli) ===")
    return hf_pipeline(
        "zero-shot-classification",
        model=MODEL_TOPIC_NLI,
        tokenizer=MODEL_TOPIC_NLI,
        use_fast=False
    )

# -------------------------
# MODEL WRAPPERS (SENTIMENT)
# -------------------------

def get_probs_finbert(clf, text: str) -> list:
    res = clf(text)[0]
    scores = {r["label"].lower(): float(r["score"]) for r in res}
    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    neu = scores.get("neutral", 0.0)
    label_3 = max(
        ("positive", "negative", "neutral"),
        key=lambda x: {"positive": pos, "negative": neg, "neutral": neu}[x]
    )
    score_3 = {"positive": pos, "negative": neg, "neutral": neu}[label_3]
    return normalize_probs(map_3class_to_5class(label_3, score_3))

def get_probs_cardiff(clf, text: str) -> list:
    res = clf(text)[0]
    scores = {r["label"].lower(): float(r["score"]) for r in res}
    neg = scores.get("negative", 0.0)
    neu = scores.get("neutral", 0.0)
    pos = scores.get("positive", 0.0)
    label_3 = max(
        ("positive", "negative", "neutral"),
        key=lambda x: {"positive": pos, "negative": neg, "neutral": neu}[x]
    )
    score_3 = {"positive": pos, "negative": neg, "neutral": neu}[label_3]
    return normalize_probs(map_3class_to_5class(label_3, score_3))

def get_probs_distilfin(clf, text: str) -> list:
    res = clf(text)[0]
    scores = {r["label"].lower(): float(r["score"]) for r in res}
    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    neu = scores.get("neutral", 0.0)
    label_3 = max(
        ("positive", "negative", "neutral"),
        key=lambda x: {"positive": pos, "negative": neg, "neutral": neu}[x]
    )
    score_3 = {"positive": pos, "negative": neg, "neutral": neu}[label_3]
    return normalize_probs(map_3class_to_5class(label_3, score_3))

# -------------------------
# DYNAMIC WEIGHTING (3-MODEL ENSEMBLE)
# -------------------------

def compute_article_dynamic_weights(model_probs_article: dict) -> dict:
    any_model = next(iter(model_probs_article.keys()))
    n_sent = len(model_probs_article[any_model])
    if n_sent == 0:
        return {m: 1.0 / len(model_probs_article) for m in model_probs_article}

    agreements = {m: 0 for m in model_probs_article}

    for i in range(n_sent):
        preds = {}
        for m, probs_list in model_probs_article.items():
            preds[m] = majority_label_from_probs(probs_list[i])
        majority_idx = Counter(preds.values()).most_common(1)[0][0]
        for m, pidx in preds.items():
            if pidx == majority_idx:
                agreements[m] += 1

    total = sum(agreements.values())
    if total <= 0:
        return {m: 1.0 / len(model_probs_article) for m in model_probs_article}

    return {m: agreements[m] / total for m in model_probs_article}

# -------------------------
# SENTIMENT PIPELINE
# -------------------------

def run_sentiment(master: dict) -> dict:
    print("=== RUNNING SENTIMENT (5-CLASS ENSEMBLE, 3 MODELS) ===")

    pipes = load_sentiment_pipelines()

    all_sentences = []
    article_weights = {}
    global_counts_5 = Counter()
    sentence_global_index = 0

    article_index_map = {a["article_id"]: a.get("article_index", 0) for a in master.get("articles", [])}

    for article in master.get("articles", []):
        article_id = article["article_id"]
        article_filename = article["article_filename"]
        body = article["body"]

        print(f"Processing article: {article_filename}")

        sents = [s.strip() for s in sent_tokenize(body) if s.strip()]

        model_probs_article = {m: [] for m in SENTIMENT_MODELS}

        for s in sents:
            model_probs_article["finbert"].append(get_probs_finbert(pipes["finbert"], s))
            model_probs_article["cardiff"].append(get_probs_cardiff(pipes["cardiff"], s))
            model_probs_article["distilfin"].append(get_probs_distilfin(pipes["distilfin"], s))

        weights = compute_article_dynamic_weights(model_probs_article)
        article_weights[article_id] = weights

        # Helper: check if sentence is mostly numeric
        def is_mostly_numeric(text):
            numbers = re.findall(r"[\d,.%]+", text)
            return len(numbers) > 0 and len(numbers) * 4 > len(text.split())

        # Helper: check if all model scores are close
        def all_scores_close(probs, threshold=0.2):
            return max(probs) - min(probs) < threshold

        # Helper: check for strong sentiment words
        STRONG_POS = {"record", "surge", "growth", "outperform", "beat", "exceed", "strong", "rise", "soar", "jump", "increase", "improve", "gain", "boost", "expand", "highest", "best"}
        STRONG_NEG = {"disappoint", "miss", "fall", "decline", "drop", "dip", "concern", "worry", "pressure", "loss", "weak", "lowest", "worst", "cut", "reduce", "slump", "plunge", "down"}
        def has_strong_sentiment(text):
            t = text.lower()
            return any(w in t for w in STRONG_POS) or any(w in t for w in STRONG_NEG)

        for idx, s in enumerate(sents):
            # Ensure sentence is a string (avoid floats/NaN from tokenization issues)
            if not isinstance(s, str):
                s = "" if s is None else str(s)
            final_probs = [0.0] * 5
            per_model_outputs = {}

            for m in SENTIMENT_MODELS:
                probs = model_probs_article[m][idx]
                w = weights[m]
                for j in range(5):
                    final_probs[j] += w * probs[j]
                per_model_outputs[m] = {
                    "probs_5": probs,
                    "pred_label_5": SENTIMENT_LABELS_5[majority_label_from_probs(probs)],
                }

            final_probs = normalize_probs(final_probs)
            final_idx = majority_label_from_probs(final_probs)
            final_label_5 = SENTIMENT_LABELS_5[final_idx]
            final_score = final_probs[final_idx]

            # 1. Only assign 'very' classes if max model confidence >0.8
            if final_label_5 in ("very_positive", "very_negative") and final_score <= 0.8:
                # Exclude both 'very' classes and renormalize so probs match chosen label
                temp_probs = final_probs[:]
                temp_probs[0] = 0.0
                temp_probs[4] = 0.0
                total = sum(temp_probs)
                if total > 0:
                    final_probs = [p / total for p in temp_probs]
                else:
                    final_probs = normalize_probs(temp_probs)
                final_idx = majority_label_from_probs(final_probs)
                final_label_5 = SENTIMENT_LABELS_5[final_idx]
                final_score = final_probs[final_idx]

            # 2. If all model scores are close (difference <0.2), assign neutral
            if all_scores_close(final_probs):
                final_label_5 = "neutral"
                final_score = final_probs[2]  # neutral index

            # 3. For sentences with contrast words, prefer main clause sentiment
            contrast_match = re.search(r"\b(but|although|however|though)\b", s, re.IGNORECASE)
            if contrast_match:
                conj = contrast_match.group(1).lower()
                parts = re.split(r"\b(but|although|however|though)\b", s, flags=re.IGNORECASE)
                if len(parts) > 2:
                    # If conjunction is 'although' or 'though', the main clause is before it.
                    # For 'but' or 'however' the main clause often comes after.
                    if conj in ("although", "though"):
                        main_clause = parts[0].strip()
                    else:
                        main_clause = parts[-1].strip()
                    # Re-run ensemble on main clause
                    clause_probs = [0.0] * 5
                    for m in SENTIMENT_MODELS:
                        if m == "finbert":
                            clause_p = get_probs_finbert(pipes[m], main_clause)
                        elif m == "cardiff":
                            clause_p = get_probs_cardiff(pipes[m], main_clause)
                        elif m == "distilfin":
                            clause_p = get_probs_distilfin(pipes[m], main_clause)
                        else:
                            continue
                        w = weights[m]
                        for j in range(5):
                            clause_probs[j] += w * clause_p[j]
                    clause_probs = normalize_probs(clause_probs)
                    clause_idx = majority_label_from_probs(clause_probs)
                    # Replace final_probs with clause_probs so saved probs match label
                    final_probs = clause_probs
                    # Re-apply 'very' class threshold logic to clause_probs
                    final_idx = clause_idx
                    final_label_5 = SENTIMENT_LABELS_5[final_idx]
                    final_score = final_probs[final_idx]
                    if final_label_5 in ("very_positive", "very_negative") and final_score <= 0.8:
                        temp_probs = final_probs[:]
                        temp_probs[0] = 0.0
                        temp_probs[4] = 0.0
                        s2 = sum(temp_probs)
                        if s2 > 0:
                            final_probs = [p / s2 for p in temp_probs]
                        else:
                            final_probs = normalize_probs(temp_probs)
                        final_idx = majority_label_from_probs(final_probs)
                        final_label_5 = SENTIMENT_LABELS_5[final_idx]
                        final_score = final_probs[final_idx]
                    if all_scores_close(final_probs):
                        final_label_5 = "neutral"
                        final_score = final_probs[2]

            # 4. If sentence is mostly numeric and lacks strong sentiment words, bias toward neutral
            if is_mostly_numeric(s) and not has_strong_sentiment(s):
                final_label_5 = "neutral"
                final_score = final_probs[2]

            if final_label_5 in ["very_negative", "negative"]:
                label_3 = "negative"
            elif final_label_5 in ["very_positive", "positive"]:
                label_3 = "positive"
            else:
                label_3 = "neutral"

            global_counts_5[final_label_5] += 1

            all_sentences.append({
                "global_index": int(sentence_global_index),
                "sentence_index_article": int(idx),
                "sentence": s,
                "article_id": article_id,
                "article_filename": article_filename,
                "article_index": article_index_map.get(article_id, 0),
                "label_5": final_label_5,
                "label": label_3,
                "score": float(final_score),
                "probs_5": [float(p) for p in final_probs],
                "model_outputs": per_model_outputs,
            })
            sentence_global_index += 1
            sentence_global_index += 1

    master["sentences"] = all_sentences

    total = sum(global_counts_5.values())
    master["sentiment_5"] = {k: v / total for k, v in global_counts_5.items()} if total > 0 else {}

    master["article_weights"] = article_weights
    return master

# -------------------------
# SENTENCE CONTEXT FEATURES (journalism-aware)
# -------------------------

HEDGING_PATTERN = re.compile(r"\b(may|might|could|potentially|possibly|appears to|seems to)\b", re.IGNORECASE)
ATTRIBUTION_PATTERN = re.compile(r"\b(analysts? said|according to|people familiar with the matter|sources said)\b", re.IGNORECASE)
CONTRAST_PATTERN = re.compile(r"\b(despite|although|even though|however|but)\b", re.IGNORECASE)
FORECAST_PATTERN = re.compile(r"\b(is expected to|is forecast to|is projected to|outlook|guidance)\b", re.IGNORECASE)

def extract_context_features(text: str) -> dict:
    return {
        "has_hedging": bool(HEDGING_PATTERN.search(text)),
        "has_attribution": bool(ATTRIBUTION_PATTERN.search(text)),
        "has_contrast": bool(CONTRAST_PATTERN.search(text)),
        "has_forecast": bool(FORECAST_PATTERN.search(text)),
    }

# -------------------------
# HYBRID TOPIC CLASSIFIER (mpnet + DeBERTa-MNLI + entity/context nudging)
# -------------------------


def run_topics_hybrid(master: dict, topic_threshold=None, alpha_embed=None, alpha_nli=None, verbose=True):
    if verbose:
        print("=== HYBRID TOPIC CLASSIFICATION (mpnet + DeBERTa-MNLI + nudging) ===")

    # Use provided or global values
    topic_threshold = topic_threshold if topic_threshold is not None else TOPIC_THRESHOLD
    alpha_embed = alpha_embed if alpha_embed is not None else TOPIC_ALPHA_EMBED
    alpha_nli = alpha_nli if alpha_nli is not None else TOPIC_ALPHA_NLI

    sentences = master.get("sentences", [])
    if not sentences:
        master["topics"] = []
        return master, None

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    topic_names = list(TOPIC_DEFINITIONS.keys())
    topic_anchor_texts = []
    topic_anchor_index = []

    for ti, tname in enumerate(topic_names):
        anchors = TOPIC_DEFINITIONS[tname]
        for a in anchors:
            topic_anchor_texts.append(a)
            topic_anchor_index.append(ti)

    anchor_embeddings = embedder.encode(topic_anchor_texts, show_progress_bar=False)

    topic_embeddings = []
    for ti in range(len(topic_names)):
        idxs = [i for i, k in enumerate(topic_anchor_index) if k == ti]
        vecs = anchor_embeddings[idxs]
        topic_embeddings.append(np.mean(vecs, axis=0))
    topic_embeddings = np.vstack(topic_embeddings)

    sent_texts = [s["sentence"] for s in sentences]
    sent_embeddings = embedder.encode(sent_texts, show_progress_bar=True)

    sims_matrix = cosine_similarity(sent_embeddings, topic_embeddings)
    emb_scores = np.apply_along_axis(lambda x: normalize_probs(list(x)), 1, sims_matrix)

    topic_nli = load_topic_nli_pipeline()
    candidate_labels = topic_names

    nli_scores = []
    if verbose:
        print("=== RUNNING DeBERTa-MNLI FOR TOPICS ===")
    for text in sent_texts:
        try:
            res = topic_nli(text, candidate_labels=candidate_labels, multi_label=False)
            scores = res.get("scores", [])
            labels = res.get("labels", [])
            score_vec = [0.0] * len(topic_names)
            for label, sc in zip(labels, scores):
                if label in candidate_labels:
                    j = candidate_labels.index(label)
                    score_vec[j] = float(sc)
            score_vec = normalize_probs(score_vec)
        except Exception:
            score_vec = [1.0 / len(topic_names)] * len(topic_names)
        nli_scores.append(score_vec)

    nli_scores = np.array(nli_scores)

    hybrid_scores = alpha_embed * emb_scores + alpha_nli * nli_scores

    # entity-aware topic hints
    def entity_topic_hint(sent):
        hints = Counter()
        ents = sent.get("entities", [])
        for e in ents or []:
            label = e.get("label", "")
            text = e.get("text", "").lower()

            if label in {"ORG", "NORP"} and any(x in text for x in ["authority", "regulator", "commission", "fca", "pra"]):
                hints["Regulation & Compliance"] += 1

            if label == "PERSON":
                hints["Leadership"] += 1

            if label == "CAP_PHRASE" and any(x in text for x in ["transformation", "strategy", "programme", "initiative"]):
                hints["Strategy & Transformation"] += 1

            if label == "FIN_TERM":
                if any(x in text for x in ["profit", "earnings", "revenue", "loss", "solvency", "capital", "guidance"]):
                    hints["Financial Performance & Market Position"] += 1

            if label == "FIN_TERM" and any(x in text for x in ["claims", "premium", "customer", "policyholder"]):
                hints["Customer Experience & Service Delivery"] += 1

        return hints

    # apply nudging + context and assign
    for i, s in enumerate(sentences):
        base_row = hybrid_scores[i].copy()

        ctx = extract_context_features(s["sentence"])
        s["context_features"] = ctx

        ctx_boost = np.zeros_like(base_row)

        if ctx["has_forecast"]:
            if "Financial Performance & Market Position" in topic_names:
                j = topic_names.index("Financial Performance & Market Position")
                ctx_boost[j] += 0.05

        if ctx["has_attribution"] or ctx["has_contrast"]:
            if "Corporate Reputation & Public Perception" in topic_names:
                j = topic_names.index("Corporate Reputation & Public Perception")
                ctx_boost[j] += 0.03

        ent_hints = entity_topic_hint(s)
        ent_boost = np.zeros_like(base_row)
        for tname, cnt in ent_hints.items():
            if tname in topic_names:
                j = topic_names.index(tname)
                ent_boost[j] += 0.04 * min(cnt, 3)

        row = base_row + ctx_boost + ent_boost
        row = normalize_probs(list(row))
        row = np.array(row)

        best_idx = int(np.argmax(row))
        best_score = float(row[best_idx])

        s["topic_scores_embedding"] = [float(x) for x in emb_scores[i]]
        s["topic_scores_nli"] = [float(x) for x in nli_scores[i]]
        s["topic_scores_hybrid"] = [float(x) for x in row]

        if best_score >= topic_threshold:
            s["topic_name"] = topic_names[best_idx]
            s["topic_score"] = best_score
        else:
            s["topic_name"] = "None"
            s["topic_score"] = best_score

    # article-level smoothing and smart fallback
    by_article = {}
    for s in sentences:
        by_article.setdefault(s["article_id"], []).append(s)

    for aid, slist in by_article.items():
        slist.sort(key=lambda x: x["sentence_index_article"])
        for idx, s in enumerate(slist):
            if s["topic_name"] != "None":
                continue

            max_score = max(s["topic_scores_hybrid"])
            if max_score < 0.15:
                continue

            neighbours = []
            if idx > 0:
                neighbours.append(slist[idx - 1])
            if idx < len(slist) - 1:
                neighbours.append(slist[idx + 1])

            neighbour_topics = [n["topic_name"] for n in neighbours if n.get("topic_name") not in (None, "None")]
            if neighbour_topics:
                cand = Counter(neighbour_topics).most_common(1)[0][0]
                s["topic_name"] = cand
                s["topic_score"] = max_score

    topic_summary = []
    for t in topic_names:
        count = sum(1 for s in sentences if s.get("topic_name") == t)
        definition_text = " ".join(TOPIC_DEFINITIONS[t])
        topic_summary.append({
            "topic_name": t,
            "size": count,
            "definition": definition_text,
        })

    none_count = sum(1 for s in sentences if s.get("topic_name") == "None")
    topic_summary.append({
        "topic_name": "None",
        "size": none_count,
        "definition": "Sentences that do not strongly match any predefined topic.",
    })

    master["topics"] = topic_summary
    master["sentences"] = sentences
    return master, sent_embeddings




# -------------------------
# ENTITY EXTRACTION (GENERIC + CAPITALISED + FINANCIAL KEYWORDS)
# -------------------------

CAPITALISED_STOPWORDS = {
    "The", "A", "An", "And", "Or", "Of", "In", "On", "At", "For",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"
}

CAPITALISED_PATTERN = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9&]+(?:\s+[A-Z][a-zA-Z0-9&]+){0,4})\b"
)

FINANCIAL_KEYWORDS = [
    "solvency", "solvency ratio", "capital ratio", "dividend", "payout",
    "earnings", "revenue", "profit", "loss", "valuation", "market share",
    "premium", "claims", "annuities", "bulk purchase annuity", "BPA",
    "fund", "investment", "portfolio", "assets", "liabilities",
    "guidance", "forecast", "outlook", "Q1", "Q2", "Q3", "Q4",
    "regulator", "FCA", "PRA", "IFRS", "Solvency II"
]

FUND_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]+ (Fund|Trust|ETF)\b")
CODE_PATTERN = re.compile(r"\b[A-Z]{2,5}\d{1,4}\b")
INDEX_PATTERN = re.compile(r"\bFTSE\s*\d{3,4}\b", re.IGNORECASE)

def extract_capitalised_phrases(text: str) -> list:
    candidates = CAPITALISED_PATTERN.findall(text)
    clean = []
    for c in candidates:
        tokens = c.split()
        if any(t in CAPITALISED_STOPWORDS for t in tokens):
            continue
        if len(c) < 3:
            continue
        clean.append(c)
    return clean

def extract_financial_keywords(text: str) -> list:
    found = []
    lower = text.lower()

    for kw in FINANCIAL_KEYWORDS:
        if kw.lower() in lower:
            found.append({
                "text": kw,
                "label": "FIN_TERM",
                "score": 1.0,
                "source": "financial_keyword",
                "start": -1,
                "end": -1,
            })

    for m in FUND_PATTERN.findall(text):
        found.append({
            "text": m,
            "label": "FIN_FUND",
            "score": 1.0,
            "source": "financial_regex",
            "start": -1,
            "end": -1,
        })

    for m in CODE_PATTERN.findall(text):
        found.append({
            "text": m,
            "label": "FIN_CODE",
            "score": 1.0,
            "source": "financial_regex",
            "start": -1,
            "end": -1,
        })

    for m in INDEX_PATTERN.findall(text):
        found.append({
            "text": m,
            "label": "FIN_INDEX",
            "score": 1.0,
            "source": "financial_regex",
            "start": -1,
            "end": -1,
        })

    return found

def normalize_entity_text_for_linking(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\b(plc|PLC|Inc|Ltd|LLC)\b\.?", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.lower().strip()

def run_entity_extraction(master: dict) -> dict:
    print("=== RUNNING ENTITY EXTRACTION (GENERIC + CAPITALISED + FINANCIAL KEYWORDS) ===")

    gen_ner = hf_pipeline("ner", model=MODEL_GENERIC_NER, aggregation_strategy="simple")

    sentences = master.get("sentences", [])
    corpus_entities = []

    for s in sentences:
        text = s["sentence"]

        gen_ents = gen_ner(text)
        capitalised = extract_capitalised_phrases(text)
        fin_terms = extract_financial_keywords(text)

        sent_entities = []

        for e in gen_ents:
            ent = {
                "text": e["word"],
                "label": e["entity_group"],
                "score": float(e["score"]),
                "source": "generic_ner",
                "start": int(e["start"]),
                "end": int(e["end"]),
            }
            sent_entities.append(ent)
            corpus_entities.append({**ent, "sentence_label_5": s["label_5"], "article_index": s.get("article_index", 0)})

        for c in capitalised:
            ent = {
                "text": c,
                "label": "CAP_PHRASE",
                "score": 1.0,
                "source": "capitalised",
                "start": -1,
                "end": -1,
            }
            sent_entities.append(ent)
            corpus_entities.append({**ent, "sentence_label_5": s["label_5"], "article_index": s.get("article_index", 0)})

        for f in fin_terms:
            sent_entities.append(f)
            corpus_entities.append({**f, "sentence_label_5": s["label_5"], "article_index": s.get("article_index", 0)})

        s["entities"] = sent_entities

    master["sentences"] = sentences

    if corpus_entities:
        df = pd.DataFrame(corpus_entities)

        counts = (
            df.groupby(["label", "text"])
              .agg(count=("text", "size"), avg_score=("score", "mean"))
              .reset_index()
              .sort_values("count", ascending=False)
        )
        master["entities_corpus"] = [
            {
                "label": str(row["label"]),
                "text": str(row["text"]),
                "count": int(row["count"]),
                "avg_score": float(row["avg_score"])
            }
            for _, row in counts.iterrows()
        ]

        df["canonical"] = df["text"].apply(normalize_entity_text_for_linking)
        link_stats = (
            df.groupby("canonical")
              .agg(
                  total_count=("text", "size"),
                  labels=("label", lambda x: list(sorted(set(x)))),
                  variants=("text", lambda x: list(sorted(set(x))))
              )
              .reset_index()
              .sort_values("total_count", ascending=False)
        )
        master["entities_linked"] = [
            {
                "canonical": str(row["canonical"]),
                "total_count": int(row["total_count"]),
                "labels": list(row["labels"]),
                "variants": list(row["variants"]),
            }
            for _, row in link_stats.iterrows()
        ]

        sentiment_weight = {
            "very_positive": 2,
            "positive": 1,
            "neutral": 0,
            "negative": -1,
            "very_negative": -2,
        }
        df["sentiment_weight"] = df["sentence_label_5"].map(sentiment_weight).astype(float)

        ent_sent = (
            df.groupby("text")
              .agg(
                  count=("text", "size"),
                  avg_sentiment=("sentiment_weight", "mean"),
              )
              .reset_index()
              .sort_values("count", ascending=False)
        )
        master["entity_sentiment"] = [
            {
                "text": str(row["text"]),
                "count": int(row["count"]),
                "avg_sentiment_weight": float(row["avg_sentiment"]),
            }
            for _, row in ent_sent.iterrows()
        ]

        ent_time = (
            df.groupby(["text", "article_index"])
              .size()
              .reset_index(name="count")
        )
        master["entity_timeline"] = [
            {
                "text": str(row["text"]),
                "article_index": int(row["article_index"]),
                "count": int(row["count"]),
            }
            for _, row in ent_time.sort_values(["text", "article_index"]).iterrows()
        ]

    else:
        master["entities_corpus"] = []
        master["entities_linked"] = []
        master["entity_sentiment"] = []
        master["entity_timeline"] = []

    return master

# -------------------------
# TOPIC SUMMARIES
# -------------------------

def build_topic_summaries(master: dict) -> dict:
    print("=== BUILDING TOPIC SUMMARIES ===")
    topics = master.get("topics", [])
    sentences = master.get("sentences", [])
    if not topics or not sentences:
        return master

    df_sent = pd.DataFrame(sentences)

    for topic in topics:
        tname = topic["topic_name"]
        topic_sentences = df_sent[df_sent["topic_name"] == tname]

        if not topic_sentences.empty:
            topic_sentences_sorted = topic_sentences.reindex(
                topic_sentences["score"].abs().sort_values(ascending=False).index
            )
            rep_sentences = topic_sentences_sorted["sentence"].head(3).tolist()
        else:
            rep_sentences = []

        topic_ents = []
        if not topic_sentences.empty and "entities" in topic_sentences.columns:
            ents_flat = []
            for ents in topic_sentences["entities"]:
                if isinstance(ents, list):
                    ents_flat.extend(ents)
            if ents_flat:
                df_e = pd.DataFrame(ents_flat)
                topic_ents = (
                    df_e.groupby(["label", "text"])
                        .size()
                        .reset_index(name="count")
                        .sort_values("count", ascending=False)
                        .head(5)["text"]
                        .tolist()
                )

        summary_parts = []

        definition = topic.get("definition")
        if isinstance(definition, list):
            definition_text = " ".join(definition)
        else:
            definition_text = str(definition)
        summary_parts.append(definition_text)

        if rep_sentences:
            summary_parts.append("Representative statements: " + " ".join(rep_sentences))
        if topic_ents:
            summary_parts.append("Key entities: " + ", ".join(topic_ents) + ".")

        topic["summary"] = " ".join(summary_parts)

    master["topics"] = topics
    return master

# -------------------------
# PLOTS
# -------------------------

def plot_sentiment(master: dict):
    print("=== PLOTTING SENTIMENT (5-CLASS) ===")
    dist = master.get("sentiment_5", {})
    if not dist:
        print("No sentiment data to plot.")
        return

    labels = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    values = [dist.get(l, 0) for l in labels]
    colors = ["darkred", "red", "grey", "green", "darkgreen"]

    plt.figure()
    plt.bar(labels, values, color=colors)
    plt.title("Sentiment Distribution (5-class)")
    plt.tight_layout()
    out = FIGURES_DIR / "sentiment_master_5class.png"
    plt.savefig(out)
    plt.close()
    print("Saved:", out)


def plot_topic_sizes(master: dict):
    print("=== PLOTTING TOPIC SIZES ===")
    topics = master.get("topics", [])
    if not topics:
        print("No topic info to plot.")
        return

    labels = [t["topic_name"] for t in topics]
    sizes = [t["size"] for t in topics]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), sizes)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title("Topic Sizes")
    plt.xlabel("Topic")
    plt.ylabel("Number of Sentences")
    plt.tight_layout()
    out = FIGURES_DIR / "topic_sizes.png"
    plt.savefig(out)
    plt.close()
    print("Saved:", out)


def plot_topic_sentiment_heatmap(master: dict):
    print("=== PLOTTING TOPIC × SENTIMENT HEATMAP (5-CLASS) ===")
    sentences = master.get("sentences", [])
    if not sentences:
        print("No sentences to plot.")
        return

    df = pd.DataFrame([
        {"topic": s.get("topic_name"), "sentiment": s.get("label_5")}
        for s in sentences
        if "topic_name" in s and "label_5" in s
    ])

    if df.empty:
        print("No topic/sentiment data.")
        return

    order = ["very_negative", "negative", "neutral", "positive", "very_positive"]

    pivot = pd.crosstab(df["topic"], df["sentiment"], normalize="index")
    pivot = pivot.reindex(columns=order, fill_value=0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="RdYlGn", linewidths=.5)
    plt.title("Topic × Sentiment Heatmap (5-class)")
    plt.tight_layout()
    out = FIGURES_DIR / "topic_sentiment_heatmap_5class.png"
    plt.savefig(out)
    plt.close()
    print("Saved:", out)


def plot_topic_weighted_bars(master: dict):
    print("=== PLOTTING WEIGHTED TOPIC BARS (5-CLASS) ===")
    sentences = master.get("sentences", [])
    if not sentences:
        print("No sentences to plot.")
        return

    df = pd.DataFrame([
        {"topic": s.get("topic_name"), "sentiment": s.get("label_5")}
        for s in sentences
        if "topic_name" in s and "label_5" in s
    ])

    if df.empty:
        print("No topic/sentiment data.")
        return

    sentiment_weight = {
        "very_positive": 2,
        "positive": 1,
        "neutral": 0,
        "negative": -1,
        "very_negative": -2
    }

    df["weight"] = df["sentiment"].map(sentiment_weight)

    topic_scores = df.groupby("topic")["weight"].mean()
    topic_counts = df["topic"].value_counts().sort_index()

    colors = []
    for t in topic_counts.index:
        score = topic_scores.get(t, 0)
        if score > 0:
            colors.append("green")
        elif score < 0:
            colors.append("red")
        else:
            colors.append("grey")

    plt.figure(figsize=(10, 5))
    plt.bar(topic_counts.index.astype(str), topic_counts.values, color=colors)
    plt.title("Topic Frequency (Colour = Average Sentiment, 5-class)")
    plt.xlabel("Topic")
    plt.ylabel("Number of Sentences")
    plt.tight_layout()
    out = FIGURES_DIR / "topic_weighted_bars_5class.png"
    plt.savefig(out)
    plt.close()
    print("Saved:", out)

# -------------------------
# MAIN
# -------------------------


def main():
    create_master_json()
    master = json.loads(MASTER_JSON.read_text(encoding="utf-8"))

    master = run_sentiment(master)
    master, embeddings = run_topics_hybrid(master)
    master = run_entity_extraction(master)
    master = build_topic_summaries(master)

    MASTER_JSON.write_text(json.dumps(master, indent=2), encoding="utf-8")

    plot_sentiment(master)
    plot_topic_sizes(master)
    plot_topic_sentiment_heatmap(master)
    plot_topic_weighted_bars(master)

    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
