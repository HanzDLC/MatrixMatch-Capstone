# matcher.py
import json
from typing import Dict, List, Optional, Tuple

import pandas as pd
from matplotlib.figure import Figure
from sentence_transformers import SentenceTransformer, util

from matrixmatch_app.db import db_cursor
from matrixmatch_app.parsers import parse_keywords

_model = None


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
            SELECT document_id, title, academic_program
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
            }
        )

    history["keywords_list"] = parse_keywords(history.get("keywords"))
    return history, matches


def build_stage2_matrix(keywords: List[str], matches: List[Dict]) -> Optional[pd.DataFrame]:
    if not keywords or not matches:
        return None

    doc_ids = [item["document_id"] for item in matches]
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
        docs = cursor.fetchall()

    abs_by_id = {row["document_id"]: row["abstract"] for row in docs}
    abstracts = [abs_by_id.get(doc_id, "") for doc_id in doc_ids]

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
