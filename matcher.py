# matcher.py
import html
import json
import re
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from matplotlib.figure import Figure
from sentence_transformers import SentenceTransformer, util

from matrixmatch_app.db import db_cursor
from matrixmatch_app.parsers import parse_keywords

_model = None
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "along",
    "also",
    "among",
    "and",
    "are",
    "because",
    "been",
    "being",
    "below",
    "between",
    "both",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "here",
    "how",
    "into",
    "its",
    "just",
    "more",
    "most",
    "other",
    "our",
    "out",
    "over",
    "same",
    "should",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "very",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "with",
    "would",
    "your",
}


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _load_documents(program_filter: str) -> List[Dict]:
    with db_cursor() as cursor:
        if program_filter and program_filter != "ALL":
            cursor.execute(
                """
                SELECT document_id, title, academic_program, abstract
                FROM documents
                WHERE academic_program = %s
                """,
                (program_filter,),
            )
        else:
            cursor.execute(
                """
                SELECT document_id, title, academic_program, abstract
                FROM documents
                """
            )
        return cursor.fetchall()


def _load_document_abstracts(doc_ids: List[int]) -> Dict[int, str]:
    if not doc_ids:
        return {}

    placeholders = ",".join(["%s"] * len(doc_ids))
    with db_cursor() as cursor:
        cursor.execute(
            f"""
            SELECT document_id, abstract
            FROM documents
            WHERE document_id IN ({placeholders})
            """,
            tuple(doc_ids),
        )
        return {row["document_id"]: row.get("abstract") or "" for row in cursor.fetchall()}


def _resolve_match_abstracts(matches: List[Dict]) -> Dict[int, str]:
    if not matches:
        return {}

    abstracts_by_doc: Dict[int, str] = {}
    missing_ids: Set[int] = set()

    for match in matches:
        doc_id = match.get("document_id")
        if doc_id is None:
            continue
        abstract = match.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            abstracts_by_doc[doc_id] = abstract
        else:
            missing_ids.add(doc_id)

    if missing_ids:
        abstracts_by_doc.update(_load_document_abstracts(list(missing_ids)))

    return abstracts_by_doc


def _split_sentences(text: str, max_sentences: int = 80) -> List[str]:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return []

    parts = [p.strip(" -") for p in _SENTENCE_SPLIT_RE.split(normalized) if p.strip()]
    if len(parts) == 1 and len(parts[0]) > 240:
        parts = [p.strip(" -") for p in re.split(r"[;:]", parts[0]) if p.strip()]

    return parts[:max_sentences]


def _extract_terms(text: str) -> Set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if len(token) > 2 and token.lower() not in _STOPWORDS
    }


def _highlight_terms(text: str, terms: Set[str]) -> str:
    highlighted = html.escape(text or "")
    if not terms:
        return highlighted

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )
    return pattern.sub(r"<mark>\1</mark>", highlighted)


def _render_sentence_overview(
    sentences: List[str],
    hit_indices: Set[int],
    terms_by_index: Dict[int, Set[str]],
) -> str:
    """Render full abstract text with sentence-level hit highlighting."""
    chunks: List[str] = []
    for idx, sentence in enumerate(sentences):
        sentence_html = _highlight_terms(sentence, terms_by_index.get(idx, set()))
        if idx in hit_indices:
            chunks.append(f'<span class="semantic-hit">{sentence_html}</span>')
        else:
            chunks.append(sentence_html)
    return " ".join(chunks)


def _build_sentence_overview_items(
    sentences: List[str],
    terms_by_index: Dict[int, Set[str]],
    pair_ids_by_index: Dict[int, Set[str]],
) -> List[Dict]:
    items: List[Dict] = []
    for idx, sentence in enumerate(sentences):
        pair_ids = sorted(pair_ids_by_index.get(idx, set()))
        items.append(
            {
                "index": idx,
                "html": _highlight_terms(sentence, terms_by_index.get(idx, set())),
                "pair_ids": pair_ids,
            }
        )
    return items


def _parse_top_matches(raw_top_matches: str) -> List[Tuple[int, float]]:
    doc_pairs: List[Tuple[int, float]] = []
    for entry in (raw_top_matches or "").split(","):
        entry = entry.strip()
        if not entry:
            continue

        if "|" in entry:
            doc_id_str, sim_str = entry.split("|", 1)
            doc_id_str = doc_id_str.strip()
            sim_str = sim_str.strip()
        else:
            doc_id_str, sim_str = entry, None

        if not doc_id_str.isdigit():
            continue

        try:
            similarity = float(sim_str) if sim_str is not None else 0.0
        except ValueError:
            similarity = 0.0
        doc_pairs.append((int(doc_id_str), similarity))

    return doc_pairs


def run_stage1(
    researcher_id: int,
    keywords: List[str],
    user_abstract: str,
    academic_program_filter: str = "ALL",
    similarity_threshold: float = 0.6,
) -> Tuple[Optional[int], List[Dict]]:
    docs = _load_documents(academic_program_filter)
    if not docs:
        return None, []

    model = get_model()
    user_emb = model.encode(user_abstract, convert_to_tensor=True)
    doc_embs = model.encode([d["abstract"] for d in docs], convert_to_tensor=True)

    sims_tensor = util.cos_sim(user_emb, doc_embs)[0]
    sims = sims_tensor.cpu().tolist()

    matches: List[Dict] = []
    for doc, sim_val in zip(docs, sims):
        if sim_val >= similarity_threshold:
            matches.append(
                {
                    "document_id": doc["document_id"],
                    "title": doc["title"],
                    "academic_program": doc["academic_program"],
                    "similarity": float(sim_val),
                }
            )

    matches.sort(key=lambda item: item["similarity"], reverse=True)

    top_matches_str = (
        ",".join(f"{m['document_id']}|{m['similarity']:.4f}" for m in matches)
        if matches
        else ""
    )
    keywords_json = json.dumps(keywords, ensure_ascii=False)

    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            INSERT INTO comparison_history
            (researcher_id, keywords, user_abstract,
             academic_program_filter, similarity_threshold, top_matches)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING history_id
            """,
            (
                researcher_id,
                keywords_json,
                user_abstract,
                academic_program_filter,
                float(similarity_threshold),
                top_matches_str,
            ),
        )
        history_id = cursor.fetchone()["history_id"]

    return history_id, matches


def run_stage2(keywords, stage1_matches, abstracts, show_heatmap=True):
    if not keywords or not stage1_matches or not abstracts:
        return None, None

    model = get_model()
    kw_embs = model.encode(keywords, convert_to_tensor=True)
    abs_embs = model.encode(abstracts, convert_to_tensor=True)
    sims = util.cos_sim(kw_embs, abs_embs).cpu().numpy()

    col_names = [f"{item[1]} (ID:{item[0]})" for item in stage1_matches]
    matrix = pd.DataFrame(sims, index=keywords, columns=col_names)
    if not show_heatmap:
        return None, matrix

    fig = Figure(figsize=(1.2 * len(col_names), 0.5 * len(keywords)))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(keywords)))
    ax.set_yticklabels(keywords, fontsize=8)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            val = matrix.iat[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{val * 100:.1f}%",
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
                fontsize=6,
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig, matrix


def get_history_with_matches(history_id):
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT
                ch.*,
                CONCAT(u.first_name, ' ', u.last_name) AS researcher_name
            FROM comparison_history ch
            JOIN "user" u ON ch.researcher_id = u.researcher_id
            WHERE ch.history_id = %s
            """,
            (history_id,),
        )
        history = cursor.fetchone()
        if not history:
            return None, []

        doc_pairs = _parse_top_matches(history.get("top_matches") or "")
        if not doc_pairs:
            history["keywords_list"] = parse_keywords(history.get("keywords"))
            return history, []

        doc_ids = [pair[0] for pair in doc_pairs]
        placeholders = ", ".join(["%s"] * len(doc_ids))
        cursor.execute(
            f"""
            SELECT document_id, title, academic_program, abstract
            FROM documents
            WHERE document_id IN ({placeholders})
            """,
            tuple(doc_ids),
        )
        docs = cursor.fetchall()

    docs_by_id = {row["document_id"]: row for row in docs}
    matches = []
    for doc_id, similarity in doc_pairs:
        doc = docs_by_id.get(doc_id)
        if not doc:
            continue
        matches.append(
                {
                    "document_id": doc["document_id"],
                    "title": doc["title"],
                    "program": doc.get("academic_program") or "",
                    "similarity": similarity,
                    "abstract": doc.get("abstract") or "",
                }
            )

    history["keywords_list"] = parse_keywords(history.get("keywords"))
    return history, matches


def build_semantic_sentence_highlights(
    user_abstract: str,
    matches: List[Dict],
    max_docs: int = 4,
    max_pairs_per_doc: int = 3,
    min_similarity: float = 0.55,
) -> List[Dict]:
    if not user_abstract or not matches:
        return []

    user_sentences = _split_sentences(user_abstract)
    if not user_sentences:
        return []

    selected_matches = matches[:max_docs]
    abstracts_by_doc = _resolve_match_abstracts(selected_matches)
    if not abstracts_by_doc:
        return []

    model = get_model()
    user_embeddings = model.encode(user_sentences, convert_to_tensor=True)

    highlights = []
    for match in selected_matches:
        doc_abstract = abstracts_by_doc.get(match["document_id"], "")
        doc_sentences = _split_sentences(doc_abstract)
        if not doc_sentences:
            continue

        doc_embeddings = model.encode(doc_sentences, convert_to_tensor=True)
        similarity_matrix = util.cos_sim(user_embeddings, doc_embeddings).cpu().numpy()

        candidates = []
        for user_idx in range(len(user_sentences)):
            doc_idx = int(similarity_matrix[user_idx].argmax())
            score = float(similarity_matrix[user_idx][doc_idx])
            if score >= min_similarity:
                candidates.append((score, user_idx, doc_idx))
        candidates.sort(key=lambda item: item[0], reverse=True)

        pairs = []
        used_user_sentences = set()
        used_doc_sentences = set()
        user_terms_by_index: Dict[int, Set[str]] = {}
        doc_terms_by_index: Dict[int, Set[str]] = {}
        user_pair_ids_by_index: Dict[int, Set[str]] = {}
        doc_pair_ids_by_index: Dict[int, Set[str]] = {}
        for score, user_idx, doc_idx in candidates:
            if user_idx in used_user_sentences or doc_idx in used_doc_sentences:
                continue

            used_user_sentences.add(user_idx)
            used_doc_sentences.add(doc_idx)

            user_sentence = user_sentences[user_idx]
            doc_sentence = doc_sentences[doc_idx]
            shared_terms = _extract_terms(user_sentence) & _extract_terms(doc_sentence)
            user_terms_by_index.setdefault(user_idx, set()).update(shared_terms)
            doc_terms_by_index.setdefault(doc_idx, set()).update(shared_terms)
            pair_id = f"pair-{match['document_id']}-{len(pairs) + 1}"
            user_pair_ids_by_index.setdefault(user_idx, set()).add(pair_id)
            doc_pair_ids_by_index.setdefault(doc_idx, set()).add(pair_id)

            pairs.append(
                {
                    "pair_id": pair_id,
                    "similarity": score,
                    "user_index": user_idx,
                    "doc_index": doc_idx,
                    "user_sentence_plain": user_sentence,
                    "doc_sentence_plain": doc_sentence,
                    "user_sentence_html": _highlight_terms(user_sentence, shared_terms),
                    "doc_sentence_html": _highlight_terms(doc_sentence, shared_terms),
                }
            )
            if len(pairs) >= max_pairs_per_doc:
                break

        if pairs:
            user_overview_html = _render_sentence_overview(
                sentences=user_sentences,
                hit_indices=used_user_sentences,
                terms_by_index=user_terms_by_index,
            )
            doc_overview_html = _render_sentence_overview(
                sentences=doc_sentences,
                hit_indices=used_doc_sentences,
                terms_by_index=doc_terms_by_index,
            )
            highlights.append(
                {
                    "document_id": match["document_id"],
                    "title": match["title"],
                    "program": match.get("program", ""),
                    "document_similarity": match.get("similarity", 0.0),
                    "pairs": pairs,
                    "user_overview_items": _build_sentence_overview_items(
                        sentences=user_sentences,
                        terms_by_index=user_terms_by_index,
                        pair_ids_by_index=user_pair_ids_by_index,
                    ),
                    "doc_overview_items": _build_sentence_overview_items(
                        sentences=doc_sentences,
                        terms_by_index=doc_terms_by_index,
                        pair_ids_by_index=doc_pair_ids_by_index,
                    ),
                    "user_overview_html": user_overview_html,
                    "doc_overview_html": doc_overview_html,
                }
            )

    return highlights


def build_stage2_matrix(keywords: List[str], matches: List[Dict]) -> Optional[pd.DataFrame]:
    if not keywords or not matches:
        return None

    doc_ids = [item["document_id"] for item in matches]
    abstracts_by_doc = _resolve_match_abstracts(matches)
    abstracts = [abstracts_by_doc.get(doc_id, "") for doc_id in doc_ids]

    model = get_model()
    kw_embs = model.encode(keywords, convert_to_tensor=True)
    abs_embs = model.encode(abstracts, convert_to_tensor=True)
    sims = util.cos_sim(kw_embs, abs_embs).cpu().numpy()

    col_names = [f"{item['title']} (ID:{item['document_id']})" for item in matches]
    return pd.DataFrame(sims, index=keywords, columns=col_names)


def build_heatmap_figure(matrix: pd.DataFrame) -> Figure:
    n_rows, n_cols = matrix.shape
    fig = Figure(figsize=(max(6, 1.2 * n_cols), max(4, 0.7 * n_rows)), dpi=100)
    ax = fig.add_subplot(111)

    im = ax.imshow(matrix.values, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=9)

    for row_index in range(n_rows):
        for col_index in range(n_cols):
            val = matrix.iat[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{val * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if val > 0.5 else "black",
            )

    ax.set_title("Stage 2 - Keyword vs Document Abstract Similarity (%)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    fig.tight_layout()
    return fig


def build_heatmap_for_history(history_id: int) -> Optional[Figure]:
    history, matches = get_history_with_matches(history_id)
    if not history or not matches:
        return None

    keywords = parse_keywords(history.get("keywords"))
    if not keywords:
        return None

    matrix = build_stage2_matrix(keywords, matches)
    if matrix is None:
        return None

    return build_heatmap_figure(matrix)
