import os
import re
import json
import base64
import time
import requests

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import easyocr
import streamlit as st
from rapidfuzz import fuzz, process


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Picky — Pick Your Book",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "bookshelf_efficientnetb0_v2.keras")
CLASS_NAMES_PATH = os.path.join(APP_DIR, "class_names.json")

WORKSPACE = "francis-infant-s-workspace"
WORKFLOW_ID = (
    "bookshelf-detector-vbookshelf-detector-"
    "yh1bv-3-rfdetr-small-t1-logic"
)

RF_CONFIDENCE_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.65
OCR_THRESHOLD_PHONE = 60
CLASSIFIER_STRONG_THRESHOLD = 0.85
CLASSIFIER_DYNAMIC_THRESHOLD = 0.75
MIN_MARGIN = 0.20


# ============================================================
# SECRET
# ============================================================

def get_roboflow_key():
    try:
        if "ROBOFLOW_API_KEY" in st.secrets:
            return st.secrets["ROBOFLOW_API_KEY"]
    except Exception:
        pass

    return os.getenv("ROBOFLOW_API_KEY")


# ============================================================
# BOOK DATA
# ============================================================

BOOK_METADATA = {
    "11 rules for life": {
        "titles": ["11 RULES FOR LIFE", "RULES FOR LIFE"],
        "authors": ["CHETAN BHAGAT"]
    },
    "a leader in the making": {
        "titles": ["A LEADER IN THE MAKING", "LEADER IN THE MAKING"],
        "authors": ["JOYCE MEYER"]
    },
    "baedeker india": {
        "titles": ["BAEDEKER INDIA", "INDIA"],
        "authors": []
    },
    "lateral thinking": {
        "titles": ["LATERAL THINKING"],
        "authors": ["EDWARD DE BONO", "DE BONO"]
    },
    "my journey": {
        "titles": ["MY JOURNEY"],
        "authors": ["A P J ABDUL KALAM", "ABDUL KALAM", "KALAM"]
    },
    "secrets of mind power": {
        "titles": ["SECRETS OF MIND POWER", "MIND POWER"],
        "authors": ["LORAYNE"]
    },
    "spirit hacking": {
        "titles": ["SPIRIT HACKING"],
        "authors": []
    },
    "the golden gate": {
        "titles": ["THE GOLDEN GATE", "GOLDEN GATE"],
        "authors": ["VIKRAM SETH", "SETH"]
    },
    "the idiot": {
        "titles": ["THE IDIOT", "IDIOT"],
        "authors": ["FYODOR DOSTOEVSKY", "DOSTOEVSKY", "FYODOR"]
    },
    "war and peace": {
        "titles": ["WAR AND PEACE"],
        "authors": ["LEO TOLSTOY", "TOLSTOY"]
    }
}

BOOK_ALIASES = {
    "11 rules for life": [
        "11 rules for life", "rules for life", "11 rules", "chetan bhagat"
    ],
    "a leader in the making": [
        "a leader in the making", "leader in the making", "leader making", "joyce meyer"
    ],
    "baedeker india": [
        "baedeker india", "baedeker", "india baedeker"
    ],
    "lateral thinking": [
        "lateral thinking", "lateral", "thinking", "edward de bono", "de bono"
    ],
    "my journey": [
        "my journey", "journey", "abdul kalam", "kalam", "apj kalam", "a p j abdul kalam"
    ],
    "secrets of mind power": [
        "secrets of mind power", "mind power", "secrets mind power", "lorayne"
    ],
    "spirit hacking": [
        "spirit hacking", "spirit", "hacking"
    ],
    "the golden gate": [
        "the golden gate", "golden gate", "golden", "vikram seth"
    ],
    "the idiot": [
        "the idiot", "idiot", "dostoevsky", "fyodor dostoevsky", "fyodor"
    ],
    "war and peace": [
        "war and peace", "war peace", "tolstoy", "leo tolstoy"
    ]
}

UNIQUE_BOOK_KEYWORDS = {
    "11 rules for life": ["RULES", "CHETAN"],
    "a leader in the making": ["LEADER", "JOYCE"],
    "baedeker india": ["BAEDEKER"],
    "lateral thinking": ["LATERAL", "DE BONO"],
    "my journey": ["JOURNEY", "KALAM"],
    "secrets of mind power": ["MIND POWER", "LORAYNE"],
    "spirit hacking": ["SPIRIT HACKING"],
    "the golden gate": ["GOLDEN GATE", "VIKRAM SETH"],
    "the idiot": ["IDIOT", "DOSTOEVSKY"],
    "war and peace": ["WAR AND PEACE", "TOLSTOY"]
}

CLASS_TO_METADATA = {
    "11 Rules For Life": "11 rules for life",
    "A Leader In The Making": "a leader in the making",
    "Baedeker India": "baedeker india",
    "Lateral Thinking": "lateral thinking",
    "My Journey": "my journey",
    "Secrets Of Mind Power": "secrets of mind power",
    "Spirit Hacking": "spirit hacking",
    "The Golden Gate": "the golden gate",
    "The Idiot": "the idiot",
    "War and Peace": "war and peace"
}

DEFAULT_CLASS_NAMES = [
    "11 Rules For Life",
    "A Leader In The Making",
    "Baedeker India",
    "Lateral Thinking",
    "My Journey",
    "Secrets Of Mind Power",
    "Spirit Hacking",
    "The Golden Gate",
    "The Idiot",
    "War and Peace"
]

DISPLAY = {
    "11 rules for life": ("11 Rules for Life", "Chetan Bhagat"),
    "a leader in the making": ("A Leader in the Making", "Joyce Meyer"),
    "baedeker india": ("Baedeker India", ""),
    "lateral thinking": ("Lateral Thinking", "Edward de Bono"),
    "my journey": ("My Journey", "A. P. J. Abdul Kalam"),
    "secrets of mind power": ("Secrets of Mind Power", "Lorayne"),
    "spirit hacking": ("Spirit Hacking", ""),
    "the golden gate": ("The Golden Gate", "Vikram Seth"),
    "the idiot": ("The Idiot", "Fyodor Dostoevsky"),
    "war and peace": ("War and Peace", "Leo Tolstoy")
}


# ============================================================
# CACHE HEAVY RESOURCES
# ============================================================

@st.cache_resource(show_spinner="Loading Picky's vision model…")
def load_classifier():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "bookshelf_efficientnetb0_v2.keras is missing from the repository."
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects={
            "preprocess_input": preprocess_input
        }
    )


@st.cache_resource(show_spinner="Loading Picky's text reader…")
def load_ocr():
    return easyocr.Reader(
        ["en"],
        gpu=False
    )


model = load_classifier()
reader = load_ocr()

input_height = int(model.input_shape[1])
input_width = int(model.input_shape[2])

if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if isinstance(class_names, dict):
        class_names = [
            class_names[str(i)] if str(i) in class_names else class_names[i]
            for i in range(len(class_names))
        ]
else:
    class_names = DEFAULT_CLASS_NAMES


# ============================================================
# TEXT / OCR HELPERS
# ============================================================

def normalize_text(text):
    if text is None:
        return ""

    text = str(text).upper()
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_ocr_text(text):
    return normalize_text(text)


STOP_WORDS = {"THE", "A", "AN", "IN", "OF", "FOR", "AND", "TO"}


def keyword_coverage(ocr_text, title):
    ocr_text = normalize_text(ocr_text)
    title = normalize_text(title)

    ocr_words = ocr_text.split()
    title_words = [
        word for word in title.split()
        if word not in STOP_WORDS
    ]

    if not title_words:
        return 0, []

    matched_words = []

    for title_word in title_words:
        best_score = 0

        for ocr_word in ocr_words:
            score = fuzz.ratio(title_word, ocr_word)
            best_score = max(best_score, score)

        if best_score >= 70:
            matched_words.append(title_word)

    coverage = (
        len(matched_words) / len(title_words)
    ) * 100

    return coverage, matched_words


def author_similarity(ocr_text, authors):
    if not authors:
        return 0

    ocr_text = normalize_text(ocr_text)
    best = 0

    for author in authors:
        author = normalize_text(author)

        score = max(
            fuzz.ratio(ocr_text, author),
            fuzz.token_set_ratio(ocr_text, author),
            fuzz.partial_ratio(ocr_text, author)
        )

        best = max(best, score)

    return best


def title_similarity(ocr_text, titles):
    ocr_text = normalize_text(ocr_text)

    best_score = 0
    best_title = None

    for title in titles:
        title_clean = normalize_text(title)

        score = max(
            fuzz.WRatio(ocr_text, title_clean),
            fuzz.token_set_ratio(ocr_text, title_clean)
        )

        if score > best_score:
            best_score = score
            best_title = title

    return best_score, best_title


def score_book_strict(ocr_text, metadata):
    phrase_score, best_title = title_similarity(
        ocr_text,
        metadata["titles"]
    )

    if best_title is None:
        return {
            "final_score": 0,
            "phrase_score": 0,
            "coverage": 0,
            "matched_words": [],
            "author_score": 0,
            "evidence": "insufficient evidence"
        }

    coverage, matched_words = keyword_coverage(
        ocr_text,
        best_title
    )

    author_score = author_similarity(
        ocr_text,
        metadata["authors"]
    )

    if coverage >= 65 and phrase_score >= 65:
        final_score = (
            0.65 * phrase_score
            + 0.35 * coverage
        )

    elif coverage >= 40 and author_score >= 80:
        final_score = (
            0.45 * phrase_score
            + 0.25 * coverage
            + 0.30 * author_score
        )

    else:
        final_score = 0

    return {
        "final_score": final_score,
        "phrase_score": phrase_score,
        "coverage": coverage,
        "matched_words": matched_words,
        "author_score": author_score
    }


# ============================================================
# QUERY RESOLVER
# ============================================================

def clean_user_query(text):
    text = str(text).lower().strip()

    remove_phrases = [
        "find me",
        "search for",
        "look for",
        "show me",
        "where is",
        "where's",
        "find",
        "search",
        "locate",
        "book",
        "please"
    ]

    for phrase in remove_phrases:
        text = text.replace(phrase, " ")

    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def resolve_book_query(user_query):
    query = clean_user_query(user_query)

    if not query:
        return None, 0

    for canonical, aliases in BOOK_ALIASES.items():
        for alias in aliases:
            if query == alias:
                return canonical, 100

    keyword_matches = []

    for canonical, aliases in BOOK_ALIASES.items():
        for alias in aliases:
            if query in alias or alias in query:
                score = max(
                    fuzz.ratio(query, alias),
                    fuzz.partial_ratio(query, alias)
                )

                keyword_matches.append(
                    (canonical, score)
                )

    if keyword_matches:
        return max(
            keyword_matches,
            key=lambda x: x[1]
        )

    alias_to_book = {}
    all_aliases = []

    for canonical, aliases in BOOK_ALIASES.items():
        for alias in aliases:
            all_aliases.append(alias)
            alias_to_book[alias] = canonical

    match = process.extractOne(
        query,
        all_aliases,
        scorer=fuzz.WRatio
    )

    if match is None:
        return None, 0

    matched_alias = match[0]
    score = float(match[1])

    if score >= 65:
        return alias_to_book[matched_alias], score

    return None, score


# ============================================================
# OCR
# ============================================================

def run_ocr_on_crop(crop):
    rotations = {
        "0": crop,
        "90": cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
        "180": cv2.rotate(crop, cv2.ROTATE_180),
        "270": cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }

    rotation_results = []

    for angle, rotated_crop in rotations.items():
        results = reader.readtext(
            rotated_crop,
            detail=1,
            paragraph=False
        )

        detected_text = []
        confidences = []

        for _, text, confidence in results:
            if confidence >= 0.15:
                detected_text.append(text)
                confidences.append(confidence)

        combined_text = " ".join(detected_text)
        cleaned_text = clean_ocr_text(
            combined_text
        )

        avg_conf = (
            float(np.mean(confidences))
            if confidences
            else 0
        )

        rotation_results.append({
            "rotation": angle,
            "text": cleaned_text,
            "confidence": avg_conf
        })

    return rotation_results


def keyword_ocr_rescue(
    ocr_text,
    requested_book
):
    text = normalize_text(ocr_text)
    keywords = UNIQUE_BOOK_KEYWORDS[
        requested_book
    ]

    best_score = 0
    best_keyword = None

    for keyword in keywords:
        keyword = normalize_text(keyword)

        if keyword in text:
            return {
                "score": 100.0,
                "keyword": keyword,
                "matched": True
            }

        score = fuzz.partial_ratio(
            text,
            keyword
        )

        if score > best_score:
            best_score = score
            best_keyword = keyword

    if best_score >= 80:
        return {
            "score": float(best_score),
            "keyword": best_keyword,
            "matched": True
        }

    return {
        "score": float(best_score),
        "keyword": best_keyword,
        "matched": False
    }


# ============================================================
# CLASSIFIER
# ============================================================

def preprocess_classifier_crop(crop):
    resized = cv2.resize(
        crop,
        (input_width, input_height),
        interpolation=cv2.INTER_AREA
    )

    image = resized.astype(
        np.float32
    )

    return np.expand_dims(
        image,
        axis=0
    )


def classify_crop(crop):
    processed = preprocess_classifier_crop(
        crop
    )

    predictions = model.predict(
        processed,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index]
    )

    predicted_class = class_names[
        predicted_index
    ]

    top_indices = np.argsort(
        predictions
    )[-3:][::-1]

    top3 = []

    for index in top_indices:
        top3.append({
            "class": class_names[index],
            "confidence": float(
                predictions[index]
            )
        })

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top3": top3
    }


# ============================================================
# NMS
# ============================================================

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_width = max(
        0,
        xi2 - xi1
    )

    inter_height = max(
        0,
        yi2 - yi1
    )

    intersection = (
        inter_width
        * inter_height
    )

    area1 = (
        (x2_1 - x1_1)
        * (y2_1 - y1_1)
    )

    area2 = (
        (x2_2 - x1_2)
        * (y2_2 - y1_2)
    )

    union = (
        area1
        + area2
        - intersection
    )

    if union == 0:
        return 0

    return intersection / union


def apply_nms(
    detections,
    iou_threshold=0.65
):
    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True
    )

    selected = []

    while detections:
        best = detections.pop(0)
        selected.append(best)

        remaining = []

        for det in detections:
            iou = calculate_iou(
                best["box"],
                det["box"]
            )

            if iou < iou_threshold:
                remaining.append(det)

        detections = remaining

    return selected


# ============================================================
# DETECTOR
# ============================================================

def run_detector(image_rgb):
    api_key = get_roboflow_key()

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY has not been configured in Streamlit Secrets."
        )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    ok, buffer = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    )

    if not ok:
        raise RuntimeError(
            "Unable to process the uploaded photo."
        )

    encoded = base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

    url = (
        "https://serverless.roboflow.com/"
        f"infer/workflows/{WORKSPACE}/{WORKFLOW_ID}"
    )

    payload = {
        "api_key": api_key,
        "inputs": {
            "image": {
                "type": "base64",
                "value": encoded
            }
        }
    }

    response = requests.post(
        url,
        json=payload,
        timeout=120
    )

    response.raise_for_status()
    result = response.json()

    try:
        return result[
            "outputs"
        ][0][
            "predictions"
        ][
            "predictions"
        ]
    except Exception:
        return []


def prepare_detections(
    image_rgb,
    predictions
):
    h, w = image_rgb.shape[:2]
    detections = []

    for index, pred in enumerate(
        predictions,
        start=1
    ):
        confidence = float(
            pred.get(
                "confidence",
                0
            )
        )

        if confidence < RF_CONFIDENCE_THRESHOLD:
            continue

        x = float(pred["x"])
        y = float(pred["y"])
        bw = float(pred["width"])
        bh = float(pred["height"])

        x1 = max(
            0,
            int(x - bw / 2)
        )

        y1 = max(
            0,
            int(y - bh / 2)
        )

        x2 = min(
            w,
            int(x + bw / 2)
        )

        y2 = min(
            h,
            int(y + bh / 2)
        )

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "index": index,
            "confidence": confidence,
            "box": (
                x1,
                y1,
                x2,
                y2
            )
        })

    return apply_nms(
        detections,
        iou_threshold=NMS_IOU_THRESHOLD
    )


def analyze_detections(
    image_rgb,
    detections
):
    analyzed = []

    for item in detections:
        x1, y1, x2, y2 = item["box"]

        crop = image_rgb[
            y1:y2,
            x1:x2
        ].copy()

        if crop.size == 0:
            continue

        analyzed.append({
            "index": item["index"],
            "confidence": item["confidence"],
            "box": item["box"],
            "crop": crop,
            "ocr_rotation_results": run_ocr_on_crop(
                crop
            ),
            "classifier_scores": classify_crop(
                crop
            )
        })

    return analyzed


# ============================================================
# FINAL LOCATOR
# ============================================================

def get_classifier_margin(
    classifier_result
):
    top1 = float(
        classifier_result.get(
            "confidence",
            0
        )
    )

    top3 = classifier_result.get(
        "top3",
        []
    )

    second_conf = None

    if len(top3) >= 2:
        second = top3[1]

        if isinstance(
            second,
            dict
        ):
            for key in [
                "confidence",
                "score",
                "probability"
            ]:
                if key in second:
                    try:
                        second_conf = float(
                            second[key]
                        )
                        break
                    except Exception:
                        pass

    if second_conf is None:
        return None, None

    margin = (
        top1
        - second_conf
    )

    return (
        second_conf,
        margin
    )


def locate_phone_book(
    phone_analyzed,
    requested_book
):
    candidates = []

    for det in phone_analyzed:
        best_ocr_score = 0.0
        best_ocr_text = ""
        best_rotation = ""
        best_keyword = None

        for rr in det[
            "ocr_rotation_results"
        ]:
            text = rr.get(
                "text",
                ""
            )

            if not text:
                continue

            score_info = score_book_strict(
                text,
                BOOK_METADATA[
                    requested_book
                ]
            )

            strict_score = float(
                score_info[
                    "final_score"
                ]
            )

            keyword_info = keyword_ocr_rescue(
                text,
                requested_book
            )

            keyword_score = 0.0

            if keyword_info[
                "matched"
            ]:
                keyword_score = float(
                    keyword_info[
                        "score"
                    ]
                )

            current_score = max(
                strict_score,
                keyword_score
            )

            if current_score > best_ocr_score:
                best_ocr_score = current_score
                best_ocr_text = text
                best_rotation = rr.get(
                    "rotation",
                    ""
                )

                if (
                    keyword_info[
                        "matched"
                    ]
                    and keyword_score
                    >= strict_score
                ):
                    best_keyword = keyword_info[
                        "keyword"
                    ]
                else:
                    best_keyword = None

        classifier_result = det[
            "classifier_scores"
        ]

        classifier_class = classifier_result[
            "predicted_class"
        ]

        classifier_conf = float(
            classifier_result[
                "confidence"
            ]
        )

        classifier_book = CLASS_TO_METADATA.get(
            classifier_class
        )

        second_conf, margin = get_classifier_margin(
            classifier_result
        )

        accepted = False
        method = None
        final_score = 0.0

        if best_ocr_score >= OCR_THRESHOLD_PHONE:
            accepted = True

            method = (
                "OCR / keyword match"
                if best_keyword
                else "OCR"
            )

            final_score = best_ocr_score

        elif (
            classifier_book == requested_book
            and classifier_conf
            >= CLASSIFIER_STRONG_THRESHOLD
        ):
            accepted = True
            method = "EfficientNet fallback"
            final_score = (
                classifier_conf
                * 100
            )

        elif (
            classifier_book == requested_book
            and classifier_conf
            >= CLASSIFIER_DYNAMIC_THRESHOLD
            and margin is not None
            and margin >= MIN_MARGIN
        ):
            accepted = True
            method = (
                "EfficientNet strong-margin fallback"
            )

            final_score = (
                classifier_conf
                * 100
            )

        if accepted:
            candidates.append({
                **det,
                "ocr_score": best_ocr_score,
                "ocr_text": best_ocr_text,
                "rotation": best_rotation,
                "keyword": best_keyword,
                "classifier_book": classifier_book,
                "classifier_confidence": classifier_conf,
                "second_confidence": second_conf,
                "classifier_margin": margin,
                "method": method,
                "final_score": final_score
            })

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda x: x[
            "final_score"
        ]
    )


# ============================================================
# DRAW RESULT
# ============================================================

def draw_result(
    image_rgb,
    result,
    requested_book
):
    output = image_rgb.copy()

    x1, y1, x2, y2 = result[
        "box"
    ]

    green = (
        18,
        170,
        83
    )

    thickness = max(
        5,
        int(
            output.shape[1]
            / 350
        )
    )

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        green,
        thickness
    )

    title = DISPLAY[
        requested_book
    ][0]

    label = (
        f"{title}  "
        f"{result['final_score']:.0f}%"
    )

    font_scale = max(
        0.75,
        min(
            1.6,
            output.shape[1]
            / 1150
        )
    )

    text_thickness = max(
        2,
        int(
            output.shape[1]
            / 550
        )
    )

    (tw, th), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_thickness
    )

    tx = max(
        4,
        min(
            x1,
            output.shape[1]
            - tw
            - 24
        )
    )

    ty = max(
        th + 24,
        y1 - 12
    )

    cv2.rectangle(
        output,
        (
            tx,
            ty - th - 18
        ),
        (
            min(
                output.shape[1] - 1,
                tx + tw + 22
            ),
            ty + baseline + 8
        ),
        green,
        -1
    )

    cv2.putText(
        output,
        label,
        (
            tx + 10,
            ty - 4
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (
            255,
            255,
            255
        ),
        text_thickness,
        cv2.LINE_AA
    )

    return output


# ============================================================
# UI CSS
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --green-950:#063D26;
        --green-900:#07522F;
        --green-700:#0F7A49;
        --green-100:#EDF8F1;
        --green-050:#F7FCF9;
        --cream:#FBFAF6;
        --ink:#1C2821;
        --muted:#6E7872;
        --line:#E4E9E6;
    }

    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(34,197,94,.05), transparent 25%),
            linear-gradient(180deg,#FFFFFF 0%,#FBFAF6 100%);
    }

    .block-container {
        max-width: 1580px;
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }

    header[data-testid="stHeader"] {
        background: rgba(255,255,255,.0);
    }

    .picky-header {
        display:grid;
        grid-template-columns:1fr auto 1fr;
        align-items:center;
        gap:24px;
        margin-bottom:18px;
    }

    .brand-wrap {
        display:flex;
        align-items:center;
        gap:14px;
    }

    .brand-icon {
        width:76px;
        height:76px;
        border-radius:18px;
        display:grid;
        place-items:center;
        background:#EEF8F1;
        border:1px solid #D9EBDD;
        font-size:42px;
    }

    .brand-name {
        font-family:Georgia,serif;
        font-size:52px;
        font-weight:800;
        color:var(--green-950);
        line-height:.95;
    }

    .brand-sub {
        font-family:Georgia,serif;
        color:#559267;
        font-size:21px;
        font-weight:700;
        margin-top:5px;
    }

    .hero-center {
        text-align:center;
    }

    .hero-title {
        font-family:Georgia,serif;
        color:var(--green-950);
        font-size:30px;
        font-weight:800;
    }

    .hero-copy {
        margin-top:5px;
        color:#656D68;
        font-size:16px;
    }

    .hero-badge {
        justify-self:end;
        padding:12px 18px;
        border-radius:14px;
        background:#EEF8F1;
        border:1px solid #D2E7D8;
        color:var(--green-900);
        font-weight:800;
        white-space:nowrap;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color:var(--line) !important;
        border-radius:20px !important;
        box-shadow:0 8px 28px rgba(35,55,45,.07);
        background:rgba(255,255,255,.97);
    }

    .section-head {
        display:flex;
        gap:11px;
        align-items:flex-start;
        margin-bottom:9px;
    }

    .step-dot {
        width:33px;
        height:33px;
        border-radius:50%;
        background:var(--green-900);
        color:white;
        display:flex;
        align-items:center;
        justify-content:center;
        font-weight:900;
        flex:0 0 auto;
    }

    .section-title {
        color:var(--green-950);
        font-size:20px;
        font-weight:850;
        line-height:1.2;
    }

    .section-copy {
        color:var(--muted);
        font-size:14px;
        margin-top:3px;
    }

    .result-head {
        display:flex;
        align-items:center;
        gap:11px;
        margin-bottom:5px;
    }

    .target-dot {
        width:42px;
        height:42px;
        border-radius:50%;
        display:flex;
        align-items:center;
        justify-content:center;
        color:white;
        background:var(--green-900);
        font-size:22px;
    }

    .result-placeholder {
        min-height:610px;
        border:1px dashed #CCD8D0;
        background:#F8FAF8;
        border-radius:16px;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        color:#7B867F;
        text-align:center;
        padding:25px;
    }

    .result-placeholder .big {
        font-size:52px;
        margin-bottom:12px;
    }

    .result-placeholder b {
        color:#47544C;
        font-size:18px;
    }

    .result-card {
        border:1px solid #CBE7D4;
        background:linear-gradient(135deg,#F6FFF8,#EFFAF2);
        border-radius:16px;
        padding:18px 20px;
        margin-top:12px;
    }

    .peek {
        color:var(--green-900);
        font-weight:900;
        font-size:18px;
    }

    .result-title {
        font-size:25px;
        font-weight:850;
        color:#1E2922;
        margin-top:5px;
    }

    .author {
        color:#748078;
        margin-top:2px;
    }

    .result-copy {
        color:#5E6A63;
        margin-top:6px;
    }

    .metric-row {
        display:flex;
        gap:12px;
        margin-top:14px;
    }

    .mini-metric {
        flex:1;
        background:white;
        border:1px solid #DDE9E1;
        border-radius:12px;
        padding:11px 13px;
    }

    .mini-metric span {
        display:block;
        color:#7A867F;
        font-size:12px;
    }

    .mini-metric b {
        color:var(--green-900);
        font-size:20px;
    }

    .pick-banner {
        margin-top:13px;
        border-radius:12px;
        background:#EAF7ED;
        color:var(--green-950);
        text-align:center;
        padding:13px;
        font-weight:900;
    }

    .note {
        margin-top:10px;
        border-radius:10px;
        border:1px solid #DDE9E1;
        background:rgba(255,255,255,.8);
        padding:9px 11px;
        color:#607067;
        font-size:13px;
    }

    .warning-card {
        border:1px solid #EED9AE;
        background:#FFF9ED;
        border-radius:16px;
        padding:18px 20px;
        margin-top:12px;
    }

    .tip {
        padding:11px 12px;
        background:#FFF9ED;
        border:1px solid #EFDDAE;
        border-radius:11px;
        color:#625942;
        font-size:13px;
        margin-top:12px;
    }

    div.stButton > button {
        border-radius:12px;
        font-weight:800;
    }

    div.stButton > button[kind="primary"] {
        background:linear-gradient(90deg,#07522F,#0B7041);
        border:none;
        color:white;
    }

    div[data-testid="stFileUploaderDropzone"] {
        min-height:245px;
        border-radius:16px;
    }

    div[data-testid="stImage"] img {
        border-radius:15px;
        max-height:780px;
        object-fit:contain;
    }

    .footer {
        margin-top:16px;
        border-top:1px solid #E7ECE8;
        padding-top:13px;
        display:flex;
        justify-content:space-between;
        color:#7D8680;
        font-size:12px;
    }

    @media (max-width: 900px) {
        .picky-header {
            grid-template-columns:1fr;
            text-align:center;
        }

        .brand-wrap {
            justify-content:center;
        }

        .hero-badge {
            justify-self:center;
        }

        .footer {
            flex-direction:column;
            text-align:center;
            gap:4px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="picky-header">
        <div class="brand-wrap">
            <div class="brand-icon">📚🔎</div>
            <div>
                <div class="brand-name">Picky</div>
                <div class="brand-sub">Pick Your Book</div>
            </div>
        </div>

        <div class="hero-center">
            <div class="hero-title">✨ You ask. Picky finds.</div>
            <div class="hero-copy">Just point, search, and pick your book!</div>
        </div>

        <div class="hero-badge">✓ Accurate · Fast · Reliable</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "result_image" not in st.session_state:
    st.session_state.result_image = None

if "result_html" not in st.session_state:
    st.session_state.result_html = None

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "runtime_seconds" not in st.session_state:
    st.session_state.runtime_seconds = None


# ============================================================
# LAYOUT
# ============================================================

left, right = st.columns(
    [0.35, 0.65],
    gap="large"
)

with left:
    with st.container(
        border=True
    ):
        st.markdown(
            """
            <div class="section-head">
                <div class="step-dot">1</div>
                <div>
                    <div class="section-title">Upload Your Bookshelf</div>
                    <div class="section-copy">
                        Upload the full shelf photo. A distant photo is fine as long as the spines are visible.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        upload = st.file_uploader(
            "Bookshelf image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],
            label_visibility="collapsed"
        )

        camera = st.camera_input(
            "Or take a photo",
            label_visibility="visible"
        )

        st.divider()

        st.markdown(
            """
            <div class="section-head">
                <div class="step-dot">2</div>
                <div>
                    <div class="section-title">Which Book Should Picky Find?</div>
                    <div class="section-copy">
                        Enter the title, part of the title, or the author's name.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        query = st.text_input(
            "Book query",
            value=st.session_state.last_query,
            placeholder="e.g. idiot, golden gat, lateral, Tolstoy…",
            label_visibility="collapsed"
        )

        search_clicked = st.button(
            "🔎 Find My Book",
            type="primary",
            use_container_width=True
        )

        st.markdown(
            "<b style='font-size:13px'>Try these:</b>",
            unsafe_allow_html=True
        )

        q1, q2, q3 = st.columns(3)

        with q1:
            if st.button(
                "idiot",
                use_container_width=True
            ):
                st.session_state.last_query = "idiot"
                st.rerun()

        with q2:
            if st.button(
                "my journey",
                use_container_width=True
            ):
                st.session_state.last_query = "my journey"
                st.rerun()

        with q3:
            if st.button(
                "golden gate",
                use_container_width=True
            ):
                st.session_state.last_query = "golden gate"
                st.rerun()

        q4, q5 = st.columns(2)

        with q4:
            if st.button(
                "lateral",
                use_container_width=True
            ):
                st.session_state.last_query = "lateral"
                st.rerun()

        with q5:
            if st.button(
                "Tolstoy",
                use_container_width=True
            ):
                st.session_state.last_query = "Tolstoy"
                st.rerun()

        st.markdown(
            """
            <div class="tip">
                💡 <b>Tip:</b> Keep the original full-resolution image.
                Picky displays the entire shelf so a distant book remains easy to locate.
            </div>
            """,
            unsafe_allow_html=True
        )


with right:
    with st.container(
        border=True
    ):
        st.markdown(
            """
            <div class="result-head">
                <div class="target-dot">◎</div>
                <div>
                    <div class="section-title">Picky's Result</div>
                    <div class="section-copy">Here's what Picky found for you.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        result_slot = st.empty()
        status_slot = st.empty()


# ============================================================
# PROCESS SEARCH
# ============================================================

source_file = camera if camera is not None else upload

if search_clicked:
    st.session_state.last_query = query

    if source_file is None:
        status_slot.markdown(
            """
            <div class="warning-card">
                <div class="peek">📷 Add a bookshelf photo first.</div>
                <div class="result-copy">
                    Upload an image or take a photo, then search again.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif not query.strip():
        status_slot.markdown(
            """
            <div class="warning-card">
                <div class="peek">🤔 Which book should Picky find?</div>
                <div class="result-copy">
                    Type a title, part of the title, or an author's name.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        requested_book, query_score = resolve_book_query(
            query
        )

        if requested_book is None:
            status_slot.markdown(
                """
                <div class="warning-card">
                    <div class="peek">🤔 Picky isn't sure which book you mean.</div>
                    <div class="result-copy">
                        Try a title, part of the title, or the author's name.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            try:
                start_time = time.time()

                bytes_data = source_file.getvalue()

                arr = np.frombuffer(
                    bytes_data,
                    np.uint8
                )

                bgr = cv2.imdecode(
                    arr,
                    cv2.IMREAD_COLOR
                )

                if bgr is None:
                    raise RuntimeError(
                        "The uploaded image could not be read."
                    )

                image_rgb = cv2.cvtColor(
                    bgr,
                    cv2.COLOR_BGR2RGB
                )

                with st.status(
                    "Picky is scanning your shelf…",
                    expanded=True
                ) as status:
                    st.write(
                        "Looking for book-shaped regions…"
                    )

                    predictions = run_detector(
                        image_rgb
                    )

                    detections = prepare_detections(
                        image_rgb,
                        predictions
                    )

                    if not detections:
                        status.update(
                            label="Picky finished searching.",
                            state="complete"
                        )

                        st.session_state.result_image = image_rgb
                        st.session_state.result_html = (
                            "NOT_FOUND",
                            requested_book,
                            query
                        )

                    else:
                        st.write(
                            "Reading the detected book spines…"
                        )

                        analyzed = analyze_detections(
                            image_rgb,
                            detections
                        )

                        st.write(
                            "Matching the shelf against your request…"
                        )

                        result = locate_phone_book(
                            analyzed,
                            requested_book
                        )

                        if result is None:
                            st.session_state.result_image = image_rgb
                            st.session_state.result_html = (
                                "NOT_FOUND",
                                requested_book,
                                query
                            )

                        else:
                            output = draw_result(
                                image_rgb,
                                result,
                                requested_book
                            )

                            st.session_state.result_image = output
                            st.session_state.result_html = (
                                "FOUND",
                                requested_book,
                                query,
                                result
                            )

                        status.update(
                            label="Picky finished searching.",
                            state="complete"
                        )

                st.session_state.runtime_seconds = (
                    time.time()
                    - start_time
                )

            except Exception as exc:
                st.session_state.result_html = (
                    "ERROR",
                    str(exc)
                )


# ============================================================
# DISPLAY RESULT
# ============================================================

if st.session_state.result_image is None:
    result_slot.markdown(
        """
        <div class="result-placeholder">
            <div class="big">📚</div>
            <b>Your full shelf will appear here.</b>
            <div style="margin-top:6px">
                Picky keeps a large result frame so books remain visible
                even when the photo is taken from farther away.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    result_slot.image(
        st.session_state.result_image,
        use_container_width=True
    )


if st.session_state.result_html is None:
    status_slot.markdown(
        """
        <div class="result-card" style="background:#FBFCFB;border-color:#E4E9E6">
            <div class="peek" style="color:#566159">📚 Ready when you are.</div>
            <div class="result-copy">
                Upload a bookshelf photo and tell Picky which book to find.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    state = st.session_state.result_html[0]

    if state == "FOUND":
        _, requested_book, original_query, result = st.session_state.result_html

        title, author = DISPLAY[
            requested_book
        ]

        author_html = (
            f"<div class='author'>{author}</div>"
            if author
            else ""
        )

        interpretation = ""

        if clean_user_query(
            original_query
        ) != clean_user_query(
            title
        ):
            interpretation = (
                "<div class='note'>"
                f"✨ Picky understood <b>{original_query}</b> "
                f"as <b>{title}</b>."
                "</div>"
            )

        runtime = st.session_state.runtime_seconds

        runtime_html = (
            f"{runtime:.1f} sec"
            if runtime is not None
            else "—"
        )

        status_slot.markdown(
            f"""
            <div class="result-card">
                <div class="peek">👀 PEEK-A-BOOK!</div>
                <div class="result-title">There you are, {title}! 📖</div>
                {author_html}
                <div class="result-copy">
                    Picky spotted it hiding on the shelf.
                </div>

                <div class="metric-row">
                    <div class="mini-metric">
                        <span>Confidence</span>
                        <b>{result['final_score']:.1f}%</b>
                    </div>

                    <div class="mini-metric">
                        <span>Search time</span>
                        <b>{runtime_html}</b>
                    </div>
                </div>

                {interpretation}

                <div class="pick-banner">
                    🎉 NOW... PICK YOUR BOOK! 📚
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif state == "NOT_FOUND":
        _, requested_book, original_query = st.session_state.result_html

        title = DISPLAY[
            requested_book
        ][0]

        status_slot.markdown(
            f"""
            <div class="warning-card">
                <div class="peek">🙈 No Peek-a-Book this time!</div>
                <div class="result-title">
                    Picky couldn't confidently spot {title}.
                </div>
                <div class="result-copy">
                    Try a clearer shelf photo, move slightly closer,
                    or make sure the spine is visible.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif state == "ERROR":
        _, message = st.session_state.result_html

        safe = (
            str(message)
            .replace("&","&amp;")
            .replace("<","&lt;")
            .replace(">","&gt;")
        )

        status_slot.markdown(
            f"""
            <div class="warning-card">
                <div class="peek">⚠️ Picky ran into a problem.</div>
                <div class="result-copy">{safe}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown(
    """
    <div class="footer">
        <span>📖 Picky — Pick Your Book</span>
        <span>❤️ Made for book lovers</span>
        <span>Always here to help you find your next read.</span>
    </div>
    """,
    unsafe_allow_html=True
)
