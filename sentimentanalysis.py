import argparse
import json
import os
from pathlib import Path
from datetime import datetime, UTC
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

PROJECT_ROOT = Path(__file__).resolve().parent
ARTICLES_BASE_DIR = Path(os.getenv("PRESSCHOICE_ARTICLES_BASE", str(PROJECT_ROOT)))
PROJECT_ARTICLES_DIR = PROJECT_ROOT / "articles"
RAW_DIR = Path(os.getenv("PRESSCHOICE_RAW_DIR", str(ARTICLES_BASE_DIR / "Aviva Articles")))
MASTER_JSON = Path(os.getenv("PRESSCHOICE_MASTER_JSON", str(ARTICLES_BASE_DIR / "master.json")))
FIGURES_DIR = Path(os.getenv("PRESSCHOICE_FIGURES_DIR", str(ARTICLES_BASE_DIR / "figures")))
FOCUS_COMPANY_NAME = "Aviva"

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

    "Leadership & Governance": [
        "This sentence focuses on the actions, decisions, or statements of the company’s senior leadership or board.",
        "This sentence reports on appointments, resignations, succession plans, or reshuffles in the executive team.",
        "This sentence examines how the company is being steered, including governance structures, oversight, or accountability.",
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
        "This sentence considers how the company is responding to regulatory pressure, scrutiny, or compliance risks."
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


# Short, natural hypothesis phrasings for zero-shot NLI. The compound display
# names ("Workforce, Culture & Operations") make poor NLI hypotheses, which
# starved rare topics of entailment probability.
TOPIC_NLI_LABELS = {
    "Corporate Reputation & Public Perception": "the company's public reputation and how it is perceived",
    "Leadership & Governance": "company executives, board decisions and corporate governance",
    "Customer Experience & Service Delivery": "customer experience, complaints and service quality",
    "Products & Offerings": "the company's products, funds and services",
    "Financial Performance & Market Position": "financial results, profits and market performance",
    "Regulation & Compliance": "regulators, regulation and compliance",
    "Strategy & Transformation": "company strategy, restructuring and transformation plans",
    "Workforce, Culture & Operations": "staff, jobs, pay, workplace culture and day-to-day operations",
}
TOPIC_NLI_HYPOTHESIS_TEMPLATE = "This text is about {}."

# Topic threshold on the final hybrid score (0-1 absolute scale, not a
# probability distribution across topics).
TOPIC_THRESHOLD = 0.35
# blending weight between embeddings and NLI
TOPIC_ALPHA_EMBED = 0.5   # embeddings
TOPIC_ALPHA_NLI = 0.5     # DeBERTa-MNLI

# Linear ramp mapping mpnet cosine similarity onto a 0-1 confidence:
# <= EMBED_SIM_FLOOR maps to 0, >= EMBED_SIM_CEIL maps to 1.
EMBED_SIM_FLOOR = 0.15
EMBED_SIM_CEIL = 0.50

# A "None" sentence may only adopt a neighbouring topic when its own hybrid
# score for that topic is at least this value AND the topic is in its top two.
NEIGHBOUR_ADOPT_MIN_SCORE = 0.25

LOW_SENTIMENT_CONFIDENCE = 0.30
LOW_TOPIC_CONFIDENCE = 0.30

# Top-two hybrid topics closer than this (probability gap) are "borderline" / near a tie.
TOPIC_NEAR_TIE_MARGIN = 0.08
# Below this margin we attach topic_drift_risk for governance review.
TOPIC_MARGIN_DRIFT = 0.10

# Boilerplate guard: a sentence must contain at least this many words with
# letters, otherwise it is bylines/headers/artifacts ("Tara O'Connor",
# "Targeted support", "Summarise") and is dropped before classification.
MIN_SUBSTANTIVE_WORDS = 4

# Zero-width and other invisible characters that leak out of docx exports.
INVISIBLE_CHARS_PATTERN = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")

# Focus-company relevance: a sentence counts as being about the focus company
# if the company (or an alias) is mentioned in the sentence itself or in the
# preceding FOCUS_RELEVANCE_WINDOW sentences; generic continuation references
# ("the firm", "the group", "we/our" in quotes) extend the lookback window.
FOCUS_RELEVANCE_WINDOW = 2
FOCUS_RELEVANCE_WINDOW_GENERIC = 4
GENERIC_COMPANY_REFERENCE_PATTERN = re.compile(
    r"\b(the (firm|company|group|mutual|business|provider|insurer|fund house)|we|our|its)\b",
    re.IGNORECASE,
)


def normalize_company_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    cleaned = re.sub(r"\s+", " ", name).strip()
    return cleaned


def expand_company_aliases(company_name: str, aliases=None) -> list[str]:
    """
    Build the list of names used for focus-company matching.

    Compound focus names such as "Phoenix & Standard Life" almost never appear
    verbatim in journalism; articles say "Phoenix", "Phoenix Group", or
    "Standard Life". Split on &, /, "and", and commas, keep the full phrase,
    then append any explicit aliases / PRESSCHOICE_COMPANY_ALIASES.
    """
    names: list[str] = []
    primary = normalize_company_name(company_name)
    if primary:
        names.append(primary)
        # Split compound display names into searchable parts.
        parts = re.split(r"\s*(?:&|/|,|\band\b)\s*", primary, flags=re.IGNORECASE)
        for part in parts:
            part = normalize_company_name(part)
            if part and part.lower() not in {n.lower() for n in names}:
                names.append(part)

    for a in (aliases or []):
        a = normalize_company_name(a)
        if a and a.lower() not in {n.lower() for n in names}:
            names.append(a)
            for part in re.split(r"\s*(?:&|/|,|\band\b)\s*", a, flags=re.IGNORECASE):
                part = normalize_company_name(part)
                if part and part.lower() not in {n.lower() for n in names}:
                    names.append(part)

    env_aliases = os.getenv("PRESSCHOICE_COMPANY_ALIASES", "")
    for a in env_aliases.split(","):
        a = normalize_company_name(a)
        if a and a.lower() not in {n.lower() for n in names}:
            names.append(a)

    return names


def configure_runtime_paths(company_name: str | None = None, raw_dir: str | None = None, master_json: str | None = None) -> None:
    """
    Configure runtime paths before pipeline execution.
    """
    global RAW_DIR, MASTER_JSON, FIGURES_DIR, FOCUS_COMPANY_NAME

    if company_name:
        FOCUS_COMPANY_NAME = normalize_company_name(company_name) or FOCUS_COMPANY_NAME

    if raw_dir:
        RAW_DIR = Path(raw_dir)
    else:
        # Prefer original base path, but auto-fallback to local project/articles if needed.
        preferred = ARTICLES_BASE_DIR / "Aviva Articles"
        fallback = PROJECT_ARTICLES_DIR
        if any(preferred.glob("*.docx")):
            RAW_DIR = preferred
        elif fallback.exists() and any(fallback.glob("*.docx")):
            RAW_DIR = fallback
        else:
            RAW_DIR = preferred

    if master_json:
        MASTER_JSON = Path(master_json)
    else:
        MASTER_JSON = Path(os.getenv("PRESSCHOICE_MASTER_JSON", str(ARTICLES_BASE_DIR / "master.json")))

    FIGURES_DIR = Path(os.getenv("PRESSCHOICE_FIGURES_DIR", str(ARTICLES_BASE_DIR / "figures")))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Keyword nudge so the hand-coded boosts are not one-sided: previously every
# boost pushed towards Leadership/Financial/Regulation/Customer/Reputation and
# none towards Workforce, so workforce sentences were systematically stolen.
WORKFORCE_KEYWORD_PATTERN = re.compile(
    r"\b(staff|employees?|workforce|headcount|redundanc\w+|lay-?offs?|hiring|"
    r"recruit\w+|union\w*|strikes?|industrial action|morale|colleagues?|"
    r"apprentice\w*|pay gap|gender pay|pay rise|working conditions|"
    r"workplace culture|hybrid working)\b",
    re.IGNORECASE,
)

LEADERSHIP_ROLE_PATTERNS = [
    ("CEO", r"\bceo\b"),
    ("Chief Executive Officer", r"\bchief executive officer\b"),
    ("Chief Executive", r"\bchief executive\b"),
    ("Chair", r"\bchair(?:man|woman)?\b"),
    ("CFO", r"\bcfo\b"),
    ("Chief Financial Officer", r"\bchief financial officer\b"),
    ("COO", r"\bcoo\b"),
    ("Chief Operating Officer", r"\bchief operating officer\b"),
    ("Leadership", r"\bleadership\b"),
    ("Management", r"\bmanagement\b"),
]

MANUAL_SENTIMENT_LABELS = {
    "very negative": "very_negative",
    "very_negative": "very_negative",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "very positive": "very_positive",
    "very_positive": "very_positive",
}

MANUAL_TOPIC_LABELS = {
    "corporate reputation & public perception": "Corporate Reputation & Public Perception",
    "corporate reputation and public perception": "Corporate Reputation & Public Perception",
    "crpp": "Corporate Reputation & Public Perception",
    "leadership & governance": "Leadership & Governance",
    "leadership and governance": "Leadership & Governance",
    "leadership": "Leadership & Governance",
    "governance": "Leadership & Governance",
    "customer experience & service delivery": "Customer Experience & Service Delivery",
    "customer experience and service delivery": "Customer Experience & Service Delivery",
    "cx": "Customer Experience & Service Delivery",
    "products & offerings": "Products & Offerings",
    "products and offerings": "Products & Offerings",
    "financial performance & market position": "Financial Performance & Market Position",
    "financial performance and market position": "Financial Performance & Market Position",
    "performance": "Financial Performance & Market Position",
    "fp": "Financial Performance & Market Position",
    "strategy & transformation": "Strategy & Transformation",
    "strategy and transformation": "Strategy & Transformation",
    "regulation & compliance": "Regulation & Compliance",
    "regulation and compliance": "Regulation & Compliance",
    "workforce, culture & operations": "Workforce, Culture & Operations",
    "workforce culture and operations": "Workforce, Culture & Operations",
    "wo": "Workforce, Culture & Operations",
    "cb": "Corporate Reputation & Public Perception",
    "gl": "Leadership & Governance",
    "ps": "Financial Performance & Market Position",
}

TOPIC_SHORT_LABELS = {
    "Corporate Reputation & Public Perception": "CB",
    "Leadership & Governance": "GL",
    "Customer Experience & Service Delivery": "CX",
    "Products & Offerings": "PO",
    "Financial Performance & Market Position": "PS",
    "Regulation & Compliance": "RC",
    "Strategy & Transformation": "ST",
    "Workforce, Culture & Operations": "WO",
}

TOPIC_BUCKET_MAP = {
    "Corporate Reputation & Public Perception": "Customer & Brand Experience",
    "Customer Experience & Service Delivery": "Customer & Brand Experience",
    "Products & Offerings": "Customer & Brand Experience",
    "Financial Performance & Market Position": "Performance & Strategy",
    "Strategy & Transformation": "Performance & Strategy",
    "Leadership & Governance": "Governance, Leadership & Accountability",
    "Regulation & Compliance": "Governance, Leadership & Accountability",
    "Workforce, Culture & Operations": "Workforce, Culture & Operations",
}

BUCKET_SHORT_CODES = {
    "Customer & Brand Experience": "CB",
    "Workforce, Culture & Operations": "WO",
    "Performance & Strategy": "PS",
    "Governance, Leadership & Accountability": "GL",
}

GOVERNANCE_KEYWORDS = (
    "governance", "board", "oversight", "committee", "compliance", "regulator",
    "fca", "pra", "watchdog", "audit", "controls", "sanction"
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_probs(values):
    arr = np.array(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return []
    # Shift to non-negative for robust normalization.
    min_v = float(arr.min())
    if min_v < 0:
        arr = arr - min_v
    total = float(arr.sum())
    if total <= 0:
        return [1.0 / len(arr)] * len(arr)
    return (arr / total).tolist()


def majority_label_from_probs(probs):
    if not probs:
        return 2  # neutral fallback
    return int(np.argmax(np.array(probs, dtype=float)))


def _safe_hf_call(pipe, text: str):
    try:
        return pipe(text, truncation=True, max_length=512)
    except TypeError:
        return pipe(text)
    except Exception:
        return []


def _map_3way_to_5way(label: str, score: float):
    """
    Map a 3-class model output onto 5 classes using the model's confidence
    as intensity. Only near-certain calls (score above ~0.975) land in the
    "very" classes; the old fixed 0.45/0.55 split made them unreachable, so
    the 5-class output was effectively 3-class. The ramp starts at 0.95
    because the financial models are poorly calibrated near the top
    (distilfin emits >0.99 on most positive calls).
    """
    l = (label or "").lower()
    s = float(score if score is not None else 0.0)
    s = min(max(s, 0.0), 1.0)
    very = min(max((s - 0.95) / 0.05, 0.0), 1.0)
    if "negative" in l:
        return normalize_probs([s * very, s * (1.0 - very), 1.0 - s, 0.0, 0.0])
    if "positive" in l:
        return normalize_probs([0.0, 0.0, 1.0 - s, s * (1.0 - very), s * very])
    return normalize_probs([0.0, (1.0 - s) / 2.0, s, (1.0 - s) / 2.0, 0.0])


def load_sentiment_pipelines():
    return {
        "finbert": hf_pipeline("text-classification", model=MODEL_FINBERT, return_all_scores=True),
        "cardiff": hf_pipeline("text-classification", model=MODEL_CARDIFF, return_all_scores=True),
        "distilfin": hf_pipeline("text-classification", model=MODEL_DISTILFIN, return_all_scores=True),
    }


def _scores_to_probs_5(results):
    if not results:
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    rows = results[0] if isinstance(results, list) and results and isinstance(results[0], list) else results
    if not rows:
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    best = max(rows, key=lambda r: float(r.get("score", 0.0)))
    label = str(best.get("label", "neutral"))
    score = float(best.get("score", 0.0))
    return _map_3way_to_5way(label, score)


def get_probs_finbert(pipe, text: str):
    return _scores_to_probs_5(_safe_hf_call(pipe, text))


def get_probs_cardiff(pipe, text: str):
    return _scores_to_probs_5(_safe_hf_call(pipe, text))


def get_probs_distilfin(pipe, text: str):
    return _scores_to_probs_5(_safe_hf_call(pipe, text))


def compute_article_dynamic_weights(model_probs_article):
    # Confidence-based but bounded and stable.
    raw = {}
    for model_name, outputs in model_probs_article.items():
        if not outputs:
            raw[model_name] = 1.0
            continue
        conf = [max(p) for p in outputs if p]
        raw[model_name] = float(np.mean(conf)) if conf else 1.0
    total = sum(raw.values()) or 1.0
    weights = {k: v / total for k, v in raw.items()}
    # Light floor to avoid model collapse.
    floor = 0.15
    weights = {k: max(floor, w) for k, w in weights.items()}
    total2 = sum(weights.values()) or 1.0
    return {k: v / total2 for k, v in weights.items()}


def load_topic_nli_pipeline():
    return hf_pipeline("zero-shot-classification", model=MODEL_TOPIC_NLI)


def extract_context_features(text: str):
    t = (text or "").lower()
    return {
        "has_forecast": any(x in t for x in ["forecast", "outlook", "guidance", "expects", "expected"]),
        "has_attribution": any(x in t for x in ["said", "according to", "stated", "reported"]),
        "has_contrast": any(x in t for x in ["however", "but", "although", "despite", "yet"]),
    }

def get_topic_short_label(topic_name: str):
    return TOPIC_SHORT_LABELS.get(topic_name, topic_name)


def get_bucket_short_code(topic_bucket: str):
    return BUCKET_SHORT_CODES.get(topic_bucket, topic_bucket or "None")


def classify_leadership_governance_subtopic(sentence_text: str):
    if not isinstance(sentence_text, str):
        return "Leadership"
    text = sentence_text.lower()
    if any(k in text for k in GOVERNANCE_KEYWORDS):
        return "Governance"
    return "Leadership"


def _sentence_key(sentence: dict):
    txt = str(sentence.get("sentence", "")).strip()
    txt = re.sub(r"\s+", " ", txt).lower()
    return (
        str(sentence.get("article_filename", "")),
        int(sentence.get("sentence_index_article", -1)),
        txt,
    )


def load_previous_overrides():
    if not MASTER_JSON.exists():
        return {}, {}
    try:
        existing = json.loads(MASTER_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}

    by_global_index = {}
    by_sentence_key = {}
    for s in existing.get("sentences", []):
        if not isinstance(s, dict):
            continue
        gi = s.get("global_index")
        if gi is not None:
            by_global_index[int(gi)] = {
                "manual_sentiment_override": s.get("manual_sentiment_override"),
                "manual_topic_override": s.get("manual_topic_override"),
            }
        key = _sentence_key(s)
        by_sentence_key[key] = {
            "manual_sentiment_override": s.get("manual_sentiment_override"),
            "manual_topic_override": s.get("manual_topic_override"),
        }
    return by_global_index, by_sentence_key


def attach_bucket_fields(sentence: dict):
    tname = sentence.get("topic_name", "None")
    tbucket = TOPIC_BUCKET_MAP.get(tname, "None")
    sentence["topic_bucket"] = tbucket
    sentence["topic_bucket_short"] = get_bucket_short_code(tbucket)
    if tname == "Leadership & Governance":
        sentence["leadership_governance_subtopic"] = classify_leadership_governance_subtopic(sentence.get("sentence", ""))
    else:
        sentence["leadership_governance_subtopic"] = None
    return sentence

def normalize_manual_sentiment_label(label: str):
    if not isinstance(label, str):
        return None
    key = label.strip().lower().replace("-", " ").replace("_", " ")
    return MANUAL_SENTIMENT_LABELS.get(key)

def normalize_manual_topic_label(label: str):
    if not isinstance(label, str):
        return None
    candidate = label.strip().lower()
    if candidate in MANUAL_TOPIC_LABELS:
        return MANUAL_TOPIC_LABELS[candidate]
    for topic_name in TOPIC_DEFINITIONS:
        if candidate == topic_name.lower():
            return topic_name
    return None

def find_leadership_roles(text: str):
    if not isinstance(text, str):
        return []
    found = []
    lower_text = text.lower()
    for role_name, pattern in LEADERSHIP_ROLE_PATTERNS:
        if re.search(pattern, lower_text):
            found.append(role_name)
    return sorted(set(found))

def apply_manual_sentiment_override(sentence: dict):
    override = sentence.get("manual_sentiment_override")
    label_5 = normalize_manual_sentiment_label(override)
    if label_5:
        sentence["label_5"] = label_5
        sentence["label"] = (
            "negative" if label_5 in {"very_negative", "negative"} else
            "positive" if label_5 in {"very_positive", "positive"} else
            "neutral"
        )
        sentence["probs_5"] = normalize_probs([
            1.0 if i == SENTIMENT_LABELS_5.index(label_5) else 0.0
            for i in range(len(SENTIMENT_LABELS_5))
        ])
        sentence["score"] = 1.0
        sentence["sentiment_confidence"] = 1.0
        sentence["needs_review"] = False
        sentence["manual_sentiment_override_applied"] = True
    return sentence


def resolve_override_topic_score(sentence: dict, topic_name: str) -> float:
    """
    Keep manual topic overrides realistic for downstream intensity charts.
    Prefer the model's hybrid score for the selected topic when available.
    """
    topic_names = list(TOPIC_DEFINITIONS.keys())
    candidate_scores = []

    # 1) Topic-specific hybrid score from model outputs (preferred).
    hybrid_scores = sentence.get("topic_scores_hybrid")
    if isinstance(hybrid_scores, list) and topic_name in topic_names:
        idx = topic_names.index(topic_name)
        if idx < len(hybrid_scores):
            try:
                candidate_scores.append(float(hybrid_scores[idx]))
            except Exception:
                pass

    # 2) Existing sentence-level topic score as fallback.
    try:
        candidate_scores.append(float(sentence.get("topic_score", 0.0)))
    except Exception:
        pass

    valid = [s for s in candidate_scores if np.isfinite(s) and s > 0]
    if valid:
        score = max(valid)
    else:
        score = float(LOW_TOPIC_CONFIDENCE)

    # Never let manual override force "absolute certainty".
    return float(min(max(score, 0.05), 0.95))


def apply_manual_topic_override(sentence: dict):
    override = sentence.get("manual_topic_override")
    topic_name = normalize_manual_topic_label(override)
    if topic_name:
        score = resolve_override_topic_score(sentence, topic_name)
        sentence["topic_name"] = topic_name
        sentence["topic_score"] = score
        sentence["topic_confidence"] = score
        sentence["needs_review"] = False
        sentence["manual_topic_override_applied"] = True
    return sentence

def update_sentence_override(
    master: dict,
    global_index: int,
    sentiment_override: str = None,
    topic_override: str = None
) -> bool:
    for sentence in master.get("sentences", []):
        if sentence.get("global_index") == global_index:
            changed = False
            if sentiment_override is not None:
                sentence["manual_sentiment_override"] = sentiment_override
                apply_manual_sentiment_override(sentence)
                changed = True
            if topic_override is not None:
                sentence["manual_topic_override"] = topic_override
                apply_manual_topic_override(sentence)
                changed = True
            if changed:
                attach_bucket_fields(sentence)
            return True
    return False


def _is_substantive_sentence(text: str) -> bool:
    words_with_letters = [w for w in text.split() if re.search(r"[A-Za-z]", w)]
    if len(words_with_letters) < MIN_SUBSTANTIVE_WORDS:
        return False
    # Newspaper chrome / CMS artifacts that look long enough to pass the
    # word-count gate but are never company content.
    chrome = re.compile(
        r"^(article continues|read our privacy|for free real.?time|"
        r"i would like to be emailed|summarise|click here to)\b",
        re.IGNORECASE,
    )
    if chrome.search(text.strip()):
        return False
    return True


def split_sentences(body: str):
    """
    Tokenize per paragraph so headlines and standfirsts (which usually lack a
    trailing full stop) do not merge into the first body sentence. Drops
    bylines, section headers, and invisible-character artifacts.
    """
    sentences = []
    for para in (body or "").split("\n"):
        para = INVISIBLE_CHARS_PATTERN.sub("", para).strip()
        if not para:
            continue
        for s in sent_tokenize(para):
            s = s.strip()
            if s and _is_substantive_sentence(s):
                sentences.append(s)
    return sentences


def build_focus_company_pattern(company_name: str, aliases=None):
    """
    Regex matching the focus company name or any expanded alias.
    Prefer longer aliases first so "Phoenix Group" wins over "Phoenix".
    Extra aliases can also be supplied via PRESSCHOICE_COMPANY_ALIASES.
    """
    names = expand_company_aliases(company_name, aliases)
    if not names:
        return None
    # Longest match first avoids partial overlaps stealing the hit.
    names = sorted(names, key=len, reverse=True)
    joined = "|".join(re.escape(n) for n in names)
    return re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)


def compute_focus_relevance(sents, focus_pattern):
    """
    Per-sentence flags: does the sentence mention the focus company, and is it
    relevant to the focus company given nearby mentions? Market-context
    sentences (e.g. gold prices) in otherwise on-topic articles get flagged
    irrelevant so they stop skewing corpus-level sentiment and topic charts.
    """
    n = len(sents)
    if focus_pattern is None:
        return [True] * n, [True] * n

    mentions = [bool(focus_pattern.search(s)) for s in sents]
    relevant = []
    for i in range(n):
        window = FOCUS_RELEVANCE_WINDOW
        if GENERIC_COMPANY_REFERENCE_PATTERN.search(sents[i]):
            window = FOCUS_RELEVANCE_WINDOW_GENERIC
        lo = max(0, i - window)
        relevant.append(any(mentions[lo:i + 1]))
    return mentions, relevant


def _read_docx_text(docx_path: Path) -> str:
    try:
        doc = Document(str(docx_path))
        parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(parts).strip()
    except Exception:
        return ""


def create_master_json(allow_empty_input: bool = False) -> dict:
    """
    Build/refresh master.json article inventory while preserving prior sentence-level
    override state so downstream reruns can reapply manual corrections.
    """
    existing = {}
    if MASTER_JSON.exists():
        try:
            existing = json.loads(MASTER_JSON.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    article_files = sorted(RAW_DIR.glob("*.docx"), key=lambda p: p.name.lower())
    articles = []
    for idx, path in enumerate(article_files):
        body = _read_docx_text(path)
        if not body:
            continue
        articles.append({
            "article_id": f"A{idx + 1:04d}",
            "article_index": int(idx),
            "article_filename": path.name,
            "source_path": str(path),
            "ingested_at": utc_now_iso(),
            "body": body,
        })

    if len(articles) == 0 and not allow_empty_input:
        raise RuntimeError(
            "No .docx articles found in input folder; aborting to avoid overwriting master.json with empty data. "
            "Set --raw-dir to the correct folder or pass --allow-empty-input to override."
        )

    master = {
        "created_at": utc_now_iso(),
        "raw_dir": str(RAW_DIR),
        "focus_company": FOCUS_COMPANY_NAME,
        "focus_company_aliases": expand_company_aliases(FOCUS_COMPANY_NAME),
        "article_count": len(articles),
        "articles": articles,
        # Preserve historical annotations until rerun steps update them.
        "sentences": existing.get("sentences", []),
        "topics": existing.get("topics", []),
        "entities_corpus": existing.get("entities_corpus", []),
        "entities_linked": existing.get("entities_linked", []),
        "entity_sentiment": existing.get("entity_sentiment", []),
        "entity_timeline": existing.get("entity_timeline", []),
        "topic_governance": existing.get("topic_governance", {}),
    }

    MASTER_JSON.write_text(json.dumps(master, indent=2), encoding="utf-8")
    print(f"Master JSON refreshed with {len(articles)} articles.")
    return master

def run_sentiment(master: dict) -> dict:
    print("=== RUNNING SENTIMENT (5-CLASS ENSEMBLE, 3 MODELS) ===")

    pipes = load_sentiment_pipelines()
    prev_by_global_index, prev_by_sentence_key = load_previous_overrides()

    focus_pattern = build_focus_company_pattern(
        master.get("focus_company", FOCUS_COMPANY_NAME),
        master.get("focus_company_aliases", []),
    )

    all_sentences = []
    article_weights = {}
    global_counts_5 = Counter()
    global_counts_5_relevant = Counter()
    sentence_global_index = 0

    article_index_map = {a["article_id"]: a.get("article_index", 0) for a in master.get("articles", [])}

    for article in master.get("articles", []):
        article_id = article["article_id"]
        article_filename = article["article_filename"]
        body = article["body"]

        print(f"Processing article: {article_filename}")

        sents = split_sentences(body)
        sent_mentions, sent_relevant = compute_focus_relevance(sents, focus_pattern)

        model_probs_article = {m: [] for m in SENTIMENT_MODELS}
        for s in sents:
            model_probs_article["finbert"].append(get_probs_finbert(pipes["finbert"], s))
            model_probs_article["cardiff"].append(get_probs_cardiff(pipes["cardiff"], s))
            model_probs_article["distilfin"].append(get_probs_distilfin(pipes["distilfin"], s))

        weights = compute_article_dynamic_weights(model_probs_article)
        article_weights[article_id] = weights

        for idx, s in enumerate(sents):
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
            final_score = float(final_probs[final_idx])

            role_tags = find_leadership_roles(s)
            review_reasons = []
            if final_score < LOW_SENTIMENT_CONFIDENCE:
                review_reasons.append("low_sentiment_confidence")
            if role_tags:
                review_reasons.append("leadership_figure")

            if final_label_5 in ["very_negative", "negative"]:
                label_3 = "negative"
            elif final_label_5 in ["very_positive", "positive"]:
                label_3 = "positive"
            else:
                label_3 = "neutral"

            global_counts_5[final_label_5] += 1
            if sent_relevant[idx]:
                global_counts_5_relevant[final_label_5] += 1

            sentence_record = {
                "global_index": int(sentence_global_index),
                "sentence_index_article": int(idx),
                "sentence": s,
                "article_id": article_id,
                "article_filename": article_filename,
                "article_index": article_index_map.get(article_id, 0),
                "mentions_focus_company": bool(sent_mentions[idx]),
                "focus_company_relevant": bool(sent_relevant[idx]),
                "label_5": final_label_5,
                "label": label_3,
                "score": float(final_score),
                "sentiment_confidence": float(final_score),
                "probs_5": [float(p) for p in final_probs],
                "model_outputs": per_model_outputs,
                "figure_roles": role_tags,
                "review_reasons": review_reasons,
                "needs_review": bool(review_reasons),
                "manual_sentiment_override": None,
                "manual_topic_override": None,
                "manual_sentiment_override_applied": False,
                "manual_topic_override_applied": False,
            }

            # Persist manual overrides across pipeline reruns.
            prev = prev_by_global_index.get(sentence_global_index)
            if prev is None:
                prev = prev_by_sentence_key.get(_sentence_key(sentence_record))
            if prev:
                sentence_record["manual_sentiment_override"] = prev.get("manual_sentiment_override")
                sentence_record["manual_topic_override"] = prev.get("manual_topic_override")

            sentence_record = apply_manual_sentiment_override(sentence_record)
            all_sentences.append(sentence_record)
            sentence_global_index += 1

    master["sentences"] = all_sentences

    # Headline sentiment distribution covers only sentences relevant to the
    # focus company; the unfiltered distribution is kept alongside it.
    total_rel = sum(global_counts_5_relevant.values())
    total_all = sum(global_counts_5.values())
    if total_rel > 0:
        master["sentiment_5"] = {k: v / total_rel for k, v in global_counts_5_relevant.items()}
    elif total_all > 0:
        master["sentiment_5"] = {k: v / total_all for k, v in global_counts_5.items()}
    else:
        master["sentiment_5"] = {}
    master["sentiment_5_all"] = {k: v / total_all for k, v in global_counts_5.items()} if total_all > 0 else {}
    master["focus_relevance_summary"] = {
        "total_sentences": int(total_all),
        "focus_relevant_sentences": int(total_rel),
        "focus_irrelevant_sentences": int(total_all - total_rel),
    }
    master["article_weights"] = article_weights
    return master

def run_topics_hybrid(master: dict, topic_threshold=None, alpha_embed=None, alpha_nli=None, verbose=True):
    if verbose:
        print("=== HYBRID TOPIC CLASSIFICATION (mpnet + DeBERTa-MNLI + nudging) ===")

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

    sent_texts = [s["sentence"] for s in sentences]
    sent_embeddings = embedder.encode(sent_texts, show_progress_bar=True)

    # Per-topic similarity = best-matching anchor (more discriminative than a
    # centroid of same-template anchors, which are mutually very similar).
    anchor_sims = cosine_similarity(sent_embeddings, anchor_embeddings)
    sims_matrix = np.zeros((len(sentences), len(topic_names)))
    for ti in range(len(topic_names)):
        idxs = [i for i, k in enumerate(topic_anchor_index) if k == ti]
        sims_matrix[:, ti] = anchor_sims[:, idxs].max(axis=1)

    # Keep the scores ABSOLUTE: map cosine similarity onto 0-1 via a fixed
    # ramp instead of normalising across topics. The old normalisation
    # flattened every row towards 1/8 and destroyed the margins.
    emb_scores = np.clip(
        (sims_matrix - EMBED_SIM_FLOOR) / (EMBED_SIM_CEIL - EMBED_SIM_FLOOR),
        0.0, 1.0,
    )

    topic_nli = load_topic_nli_pipeline()
    nli_candidate_labels = [TOPIC_NLI_LABELS[t] for t in topic_names]
    nli_label_to_topic_idx = {lbl: j for j, lbl in enumerate(nli_candidate_labels)}

    nli_scores = []
    if verbose:
        print("=== RUNNING DeBERTa-MNLI FOR TOPICS ===")
    for text in sent_texts:
        try:
            # multi_label=True yields an independent entailment probability
            # per topic (absolute 0-1), instead of a softmax across topics
            # that forces rare topics towards zero.
            res = topic_nli(
                text,
                candidate_labels=nli_candidate_labels,
                hypothesis_template=TOPIC_NLI_HYPOTHESIS_TEMPLATE,
                multi_label=True,
            )
            scores = res.get("scores", [])
            labels = res.get("labels", [])
            score_vec = [0.0] * len(topic_names)
            for label, sc in zip(labels, scores):
                j = nli_label_to_topic_idx.get(label)
                if j is not None:
                    score_vec[j] = float(sc)
        except Exception:
            score_vec = [0.0] * len(topic_names)
        nli_scores.append(score_vec)

    nli_scores = np.array(nli_scores)
    hybrid_scores = alpha_embed * emb_scores + alpha_nli * nli_scores

    def entity_topic_hint(sent):
        hints = Counter()
        ents = sent.get("entities", [])
        for e in ents or []:
            label = e.get("label", "")
            text = e.get("text", "").lower()

            if label in {"ORG", "NORP"} and any(x in text for x in ["authority", "regulator", "commission", "fca", "pra"]):
                hints["Regulation & Compliance"] += 1
            if label == "PERSON":
                hints["Leadership & Governance"] += 1
            if label == "CAP_PHRASE" and any(x in text for x in ["transformation", "strategy", "programme", "initiative"]):
                hints["Strategy & Transformation"] += 1
            if label == "FIN_TERM":
                if any(x in text for x in ["profit", "earnings", "revenue", "loss", "solvency", "capital", "guidance"]):
                    hints["Financial Performance & Market Position"] += 1
            if label == "FIN_TERM" and any(x in text for x in ["claims", "premium", "customer", "policyholder"]):
                hints["Customer Experience & Service Delivery"] += 1
        return hints

    for i, s in enumerate(sentences):
        base_row = hybrid_scores[i].copy()
        ctx = extract_context_features(s["sentence"])
        s["context_features"] = ctx

        ctx_boost = np.zeros_like(base_row)
        if ctx["has_forecast"]:
            j = topic_names.index("Financial Performance & Market Position")
            ctx_boost[j] += 0.05
        if ctx["has_attribution"] or ctx["has_contrast"]:
            j = topic_names.index("Corporate Reputation & Public Perception")
            ctx_boost[j] += 0.03

        ent_hints = entity_topic_hint(s)
        ent_boost = np.zeros_like(base_row)
        for tname, cnt in ent_hints.items():
            if tname in topic_names:
                j = topic_names.index(tname)
                ent_boost[j] += 0.04 * min(cnt, 3)

        figure_roles = find_leadership_roles(s["sentence"])
        s["figure_roles"] = figure_roles
        if figure_roles and "Leadership & Governance" in topic_names:
            j = topic_names.index("Leadership & Governance")
            ent_boost[j] += 0.08

        if WORKFORCE_KEYWORD_PATTERN.search(s["sentence"]):
            j = topic_names.index("Workforce, Culture & Operations")
            ent_boost[j] += 0.08

        # Scores stay absolute: add the nudges and clip, but do NOT
        # renormalise across topics (that flattened the distribution).
        row = np.clip(base_row + ctx_boost + ent_boost, 0.0, 1.0)
        best_idx = int(np.argmax(row))
        best_score = float(row[best_idx])
        idx_sorted = np.argsort(row)
        second_idx = int(idx_sorted[-2]) if len(row) > 1 else best_idx
        second_best = float(row[second_idx])
        margin = float(best_score - second_best)

        s["topic_scores_embedding"] = [float(x) for x in emb_scores[i]]
        s["topic_scores_nli"] = [float(x) for x in nli_scores[i]]
        s["topic_scores_hybrid"] = [float(x) for x in row]
        s["topic_confidence"] = best_score
        s["topic_margin"] = margin
        s["topic_second_best"] = topic_names[second_idx]
        s["topic_near_tie"] = bool(margin < TOPIC_NEAR_TIE_MARGIN)
        s["topic_predicted_before_override"] = topic_names[best_idx]

        if best_score >= topic_threshold:
            s["topic_name"] = topic_names[best_idx]
            s["topic_score"] = best_score
        else:
            s["topic_name"] = "None"
            s["topic_score"] = best_score

        if best_score < LOW_TOPIC_CONFIDENCE and "low_topic_confidence" not in s.get("review_reasons", []):
            s.setdefault("review_reasons", []).append("low_topic_confidence")
            s["needs_review"] = True
        if s["topic_margin"] < TOPIC_MARGIN_DRIFT and "topic_drift_risk" not in s.get("review_reasons", []):
            s.setdefault("review_reasons", []).append("topic_drift_risk")
            s["needs_review"] = True

        s = apply_manual_topic_override(s)
        s = attach_bucket_fields(s)

    by_article = {}
    for s in sentences:
        by_article.setdefault(s["article_id"], []).append(s)

    # Context fallback for unassigned sentences. Unlike the previous version,
    # a sentence only adopts a neighbouring topic when that topic is one of
    # its OWN top-two candidates with a competitive score; the old rule
    # reassigned ~34% of the corpus to whatever big topic surrounded it,
    # which is how rare topics ended up at zero.
    for aid, slist in by_article.items():
        slist.sort(key=lambda x: x["sentence_index_article"])
        for idx, s in enumerate(slist):
            if s["topic_name"] != "None":
                continue

            neighbours = []
            if idx > 0:
                neighbours.append(slist[idx - 1])
            if idx < len(slist) - 1:
                neighbours.append(slist[idx + 1])

            neighbour_topics = [n["topic_name"] for n in neighbours if n.get("topic_name") not in (None, "None")]
            if not neighbour_topics:
                continue

            cand = Counter(neighbour_topics).most_common(1)[0][0]
            cand_idx = topic_names.index(cand)
            hybrid = np.asarray(s["topic_scores_hybrid"], dtype=float)
            cand_score = float(hybrid[cand_idx])
            top_two = set(np.argsort(hybrid)[-2:].tolist())

            if cand_idx in top_two and cand_score >= NEIGHBOUR_ADOPT_MIN_SCORE:
                s["topic_name"] = cand
                s["topic_score"] = cand_score
                s.setdefault("review_reasons", []).append("neighbour_context_assignment")
                s["needs_review"] = True
                s = attach_bucket_fields(s)

    topic_summary = []
    for t in topic_names:
        count = sum(1 for s in sentences if s.get("topic_name") == t)
        count_rel = sum(
            1 for s in sentences
            if s.get("topic_name") == t and s.get("focus_company_relevant", True)
        )
        definition_text = " ".join(TOPIC_DEFINITIONS[t])
        topic_summary.append({
            "topic_name": t,
            "short_label": get_topic_short_label(t),
            "size": count,
            "size_focus_relevant": count_rel,
            "definition": definition_text,
        })

    none_count = sum(1 for s in sentences if s.get("topic_name") == "None")
    none_count_rel = sum(
        1 for s in sentences
        if s.get("topic_name") == "None" and s.get("focus_company_relevant", True)
    )
    topic_summary.append({
        "topic_name": "None",
        "short_label": "None",
        "size": none_count,
        "size_focus_relevant": none_count_rel,
        "definition": "Sentences that do not strongly match any predefined topic.",
    })

    master["topics"] = topic_summary
    master["sentences"] = sentences
    master = compute_topic_governance_metrics(master)
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


def compute_topic_governance_metrics(master: dict) -> dict:
    sentences = master.get("sentences", [])
    if not sentences:
        master["topic_governance"] = {
            "topic_purity": [],
            "bucket_distribution": [],
            "flagged_sentences": [],
            "drift_summary": {},
        }
        return master

    df = pd.DataFrame(sentences)
    if "topic_name" not in df.columns:
        df["topic_name"] = "None"
    if "topic_bucket" not in df.columns:
        df["topic_bucket"] = df["topic_name"].map(TOPIC_BUCKET_MAP).fillna("None")
    if "topic_margin" not in df.columns:
        df["topic_margin"] = np.nan
    if "manual_topic_override_applied" not in df.columns:
        df["manual_topic_override_applied"] = False

    valid = df[df["topic_name"].notna() & df["topic_name"].ne("None")].copy()

    purity_rows = []
    bucket_rows = []
    if not valid.empty:
        for topic_name, sub in valid.groupby("topic_name"):
            bucket_counts = sub["topic_bucket"].value_counts(dropna=False)
            top_bucket = bucket_counts.index[0]
            top_count = int(bucket_counts.iloc[0])
            total = int(bucket_counts.sum())
            purity = (top_count / total) * 100.0 if total else 0.0
            purity_rows.append({
                "topic_name": topic_name,
                "expected_bucket": TOPIC_BUCKET_MAP.get(topic_name, "None"),
                "dominant_bucket": top_bucket,
                "topic_purity_percent": float(purity),
                "sentence_count": total,
            })

        bucket_dist = (
            valid.groupby(["topic_bucket", "topic_name"])
            .size()
            .reset_index(name="count")
            .sort_values(["topic_bucket", "count"], ascending=[True, False])
        )
        bucket_rows = bucket_dist.to_dict(orient="records")

    if "needs_review" in df.columns:
        flagged = df[df["needs_review"] == True].copy()
    else:
        flagged = pd.DataFrame()
    flagged_rows = []
    if not flagged.empty:
        keep_cols = [
            c for c in [
                "global_index",
                "article_id",
                "sentence",
                "topic_name",
                "topic_bucket",
                "topic_margin",
                "topic_confidence",
                "sentiment_confidence",
                "review_reasons",
            ] if c in flagged.columns
        ]
        flagged_rows = flagged[keep_cols].head(500).to_dict(orient="records")

    override_count = int(df["manual_topic_override_applied"].fillna(False).sum())
    drift_count = 0
    if "review_reasons" in df.columns:
        drift_count = int(df["review_reasons"].apply(lambda x: isinstance(x, list) and "topic_drift_risk" in x).sum())

    master["topic_governance"] = {
        "topic_purity": purity_rows,
        "bucket_distribution": bucket_rows,
        "flagged_sentences": flagged_rows,
        "drift_summary": {
            "sentences_flagged_for_drift": drift_count,
            "sentences_with_manual_topic_override": override_count,
            "total_sentences": int(len(df)),
        },
    }
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
    plt.title("Sentiment Distribution (5-class, focus-company relevant)")
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
    sizes = [t.get("size_focus_relevant", t["size"]) for t in topics]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), sizes)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title("Topic Sizes (focus-company relevant)")
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
        if "topic_name" in s and "label_5" in s and s.get("focus_company_relevant", True)
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
        if "topic_name" in s and "label_5" in s and s.get("focus_company_relevant", True)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run sentiment/topic/entity pipeline.")
    parser.add_argument("--company-name", type=str, default=None, help="Focus company name (used for metadata and article folder resolution).")
    parser.add_argument("--raw-dir", type=str, default=None, help="Override article input directory containing .docx files.")
    parser.add_argument("--master-json", type=str, default=None, help="Override output master.json path.")
    parser.add_argument(
        "--allow-empty-input",
        action="store_true",
        help="Allow zero input articles (normally blocked to protect master.json).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_runtime_paths(
        company_name=args.company_name,
        raw_dir=args.raw_dir,
        master_json=args.master_json,
    )
    print(f"Focus company: {FOCUS_COMPANY_NAME}")
    print(f"Article folder: {RAW_DIR}")
    print(f"Master JSON: {MASTER_JSON}")

    create_master_json(allow_empty_input=bool(args.allow_empty_input))
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
