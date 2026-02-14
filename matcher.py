# matcher.py
import json
from typing import List, Tuple, Dict, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from matplotlib.figure import Figure

import os

# -----------------------------
# DB CONFIG – keep in sync with app.py
# -----------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "dbname": os.getenv("DB_NAME", "matrixmatch"),
    "options": os.getenv("DB_OPTIONS", "-c search_path=matrixmatch,public"),
}


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


# -----------------------------
# SBERT model (cached)
# -----------------------------
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# -----------------------------
# Stage 1: Abstract vs Documents
# -----------------------------
def run_stage1(
    researcher_id: int,
    keywords: List[str],
    user_abstract: str,
    academic_program_filter: str = "ALL",
    similarity_threshold: float = 0.6,
) -> Tuple[Optional[int], List[Dict]]:
    """
    Stage 1:
    - Load documents from DB (optionally filter by academic_program)
    - Compute similarity between user_abstract and each document abstract
    - Keep docs >= similarity_threshold
    - Save comparison_history row and return (history_id, matches)

    matches is a list of dicts:
      {
        "document_id": int,
        "title": str,
        "academic_program": str,
        "similarity": float
      }
    """
    # 1) Load docs
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if academic_program_filter and academic_program_filter != "ALL":
            cursor.execute(
                """
                SELECT document_id, title, academic_program, abstract
                FROM documents
                WHERE academic_program = %s
                """,
                (academic_program_filter,),
            )
        else:
            cursor.execute(
                """
                SELECT document_id, title, academic_program, abstract
                FROM documents
                """
            )
        docs = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    if not docs:
        return None, []

    # 2) Encode and compute similarity
    model = get_model()
    user_emb = model.encode(user_abstract, convert_to_tensor=True)
    doc_abstracts = [d["abstract"] for d in docs]
    doc_embs = model.encode(doc_abstracts, convert_to_tensor=True)

    sims_tensor = util.cos_sim(user_emb, doc_embs)[0]
    sims = sims_tensor.cpu().tolist()  # same order as docs

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

    # Sort desc by similarity
    matches.sort(key=lambda m: m["similarity"], reverse=True)

    # 3) Save history
    if not matches:
        # still save a history row so it shows up in list
        top_matches_str = ""
    else:
        # store as "docID|similarity"
        top_matches_str = ",".join(
            f"{m['document_id']}|{m['similarity']:.4f}" for m in matches
        )

    keywords_json = json.dumps(keywords, ensure_ascii=False)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
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
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return history_id, matches

def run_stage2(keywords, stage1_matches, abstracts, show_heatmap=True):
    """
    Stage 2: Keyword vs Abstract matrix.

    keywords: list of keyword strings
    stage1_matches: list of (document_id, title, program, similarity)
    abstracts: list of document abstracts in SAME ORDER as stage1_matches

    Returns (fig, matrix)
    """
    from sentence_transformers import util
    import pandas as pd
    from matplotlib.figure import Figure

    if not keywords or not stage1_matches or not abstracts:
        return None, None

    model = get_model()

    # Encode
    kw_embs = model.encode(keywords, convert_to_tensor=True)
    abs_embs = model.encode(abstracts, convert_to_tensor=True)

    sims = util.cos_sim(kw_embs, abs_embs).cpu().numpy()

    col_names = [f"{m[1]} (ID:{m[0]})" for m in stage1_matches]
    matrix = pd.DataFrame(sims, index=keywords, columns=col_names)

    if not show_heatmap:
        return None, matrix

    # Build heatmap
    fig = Figure(figsize=(1.2 * len(col_names), 0.5 * len(keywords)))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, aspect='auto', interpolation='nearest')

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(keywords)))
    ax.set_yticklabels(keywords, fontsize=8)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iat[i, j]
            ax.text(j, i, f"{val*100:.1f}%", ha='center', va='center',
                    color='white' if val > 0.5 else 'black', fontsize=6)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig, matrix

# -----------------------------
# Helpers to read history + matches from DB
# -----------------------------
# def get_history_with_matches(
#     history_id: int,
# ) -> Tuple[Optional[Dict], List[Dict]]:
#     """
#     Load one row from comparison_history + its Stage 1 matches.
#     Returns (history_row, matches_list).
#     history_row['keywords'] is converted to a Python list.
#     """
#     conn = get_db_connection()
#     cursor = conn.cursor(dictionary=True)
#     try:
#         cursor.execute(
#             """
#             SELECT *
#             FROM comparison_history
#             WHERE history_id = %s
#             """,
#             (history_id,),
#         )
#         history = cursor.fetchone()
#     finally:
#         cursor.close()
#         conn.close()
#
#     if not history:
#         return None, []
#
#     # parse keywords
#     raw_kw = history.get("keywords") or "[]"
#     try:
#         keywords = json.loads(raw_kw)
#         if not isinstance(keywords, list):
#             keywords = [str(keywords)]
#     except Exception:
#         keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]
#
#     history["keywords"] = keywords
#
#     # parse top_matches
#     raw_top = history.get("top_matches") or ""
#     if not raw_top:
#         return history, []
#
#     pairs_raw = [p for p in raw_top.split(",") if p.strip()]
#     doc_ids: List[int] = []
#     sim_map: Dict[int, float] = {}
#     for p in pairs_raw:
#         parts = p.split("|")
#         if not parts:
#             continue
#         try:
#             doc_id = int(parts[0])
#         except ValueError:
#             continue
#         sim_val = float(parts[1]) if len(parts) > 1 else 0.0
#         doc_ids.append(doc_id)
#         sim_map[doc_id] = sim_val
#
#     if not doc_ids:
#         return history, []
#
#     # fetch doc details
#     placeholders = ",".join(["%s"] * len(doc_ids))
#     conn = get_db_connection()
#     cursor = conn.cursor(dictionary=True)
#     try:
#         cursor.execute(
#             f"""
#             SELECT document_id, title, academic_program
#             FROM documents
#             WHERE document_id IN ({placeholders})
#             """,
#             tuple(doc_ids),
#         )
#         docs = cursor.fetchall()
#     finally:
#         cursor.close()
#         conn.close()
#
#     docs_by_id = {d["document_id"]: d for d in docs}
#
#     matches: List[Dict] = []
#     # preserve order of doc_ids from top_matches
#     for did in doc_ids:
#         meta = docs_by_id.get(did)
#         if not meta:
#             continue
#         matches.append(
#             {
#                 "document_id": did,
#                 "title": meta["title"],
#                 "academic_program": meta["academic_program"],
#                 "similarity": sim_map.get(did, 0.0),
#             }
#         )
#
#     return history, matches
def get_history_with_matches(history_id):
    """
    Load a single history entry and reconstruct Stage 1 matches from the DB.

    Returns:
        history: dict (row from comparison_history + researcher_name, etc.)
        matches: list[dict] with keys:
                 - document_id
                 - title
                 - program   (from documents.academic_program)
                 - similarity
    """
    # --- Connect to DB ---
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # 1) Load history row + researcher info
        cur.execute(
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
        history = cur.fetchone()
        if not history:
            return None, []

        # 2) Parse top_matches -> list of (doc_id, similarity)
        raw_top = history.get("top_matches") or ""
        doc_pairs = []  # list of (doc_id, similarity)

        # formats we expect:
        #   "41|0.9350,8|0.8912"
        #   or "41,8,12"
        for entry in raw_top.split(","):
            entry = entry.strip()
            if not entry:
                continue

            if "|" in entry:
                doc_id_str, sim_str = entry.split("|", 1)
                doc_id_str = doc_id_str.strip()
                sim_str = sim_str.strip()
            else:
                doc_id_str = entry
                sim_str = None

            if doc_id_str.isdigit():
                doc_id = int(doc_id_str)
                try:
                    similarity = float(sim_str) if sim_str is not None else 0.0
                except ValueError:
                    similarity = 0.0
                doc_pairs.append((doc_id, similarity))

        if not doc_pairs:
            return history, []

        doc_ids = [dp[0] for dp in doc_pairs]

        # 3) Load documents from `documents` table, including academic_program
        placeholders = ", ".join(["%s"] * len(doc_ids))
        cur.execute(
            f"""
            SELECT document_id, title, academic_program
            FROM documents
            WHERE document_id IN ({placeholders})
            """,
            tuple(doc_ids),
        )
        docs = cur.fetchall()

        if not docs:
            return history, []

        docs_by_id = {row["document_id"]: row for row in docs}

        # 4) Build matches list with "program" populated
        matches = []
        for doc_id, sim in doc_pairs:
            d = docs_by_id.get(doc_id)
            if not d:
                continue

            matches.append(
                {
                    "document_id": d["document_id"],
                    "title": d["title"],
                    # 🔥 THIS is what feeds {{ m.program }} in the template
                    "program": d.get("academic_program") or "",
                    "similarity": sim,
                }
            )

        return history, matches

    finally:
        cur.close()
        conn.close()


# -----------------------------
# Stage 2: Keyword vs Abstract matrix
# -----------------------------
def build_stage2_matrix(keywords: List[str], matches: List[Dict]) -> Optional[pd.DataFrame]:
    """
    Given the keywords and Stage 1 matches, build a keyword × document
    similarity matrix (as a pandas DataFrame).
    """
    if not keywords or not matches:
        return None

    doc_ids = [m["document_id"] for m in matches]

    # fetch abstracts
    placeholders = ",".join(["%s"] * len(doc_ids))
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"""
            SELECT document_id, abstract
            FROM documents
            WHERE document_id IN ({placeholders})
            """,
            tuple(doc_ids),
        )
        docs = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    abs_by_id = {d["document_id"]: d["abstract"] for d in docs}
    abstracts = [abs_by_id.get(did, "") for did in doc_ids]

    # embeddings
    model = get_model()
    kw_embs = model.encode(keywords, convert_to_tensor=True)
    abs_embs = model.encode(abstracts, convert_to_tensor=True)

    sims_tensor = util.cos_sim(kw_embs, abs_embs)
    sims = sims_tensor.cpu().numpy()  # shape: (num_keywords, num_docs)

    col_names = [f"{m['title']} (ID:{m['document_id']})" for m in matches]
    index_names = [kw for kw in keywords]

    matrix = pd.DataFrame(sims, index=index_names, columns=col_names)
    return matrix


def build_heatmap_figure(matrix: pd.DataFrame) -> Figure:
    """
    Turn the Stage 2 matrix into a Matplotlib heatmap Figure.
    """
    n_rows, n_cols = matrix.shape
    fig = Figure(
        figsize=(max(6, 1.2 * n_cols), max(4, 0.7 * n_rows)),
        dpi=100,
    )
    ax = fig.add_subplot(111)

    im = ax.imshow(matrix.values, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=9)

    # Annotate as percentage
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix.iat[i, j]
            ax.text(
                j,
                i,
                f"{val * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if val > 0.5 else "black",
            )

    ax.set_title("Stage 2 — Keyword vs Document Abstract Similarity (%)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")

    fig.tight_layout()
    return fig


def build_heatmap_for_history(history_id: int) -> Optional[Figure]:
    """
    Convenience: for a given history_id, reload keywords + matches and build a heatmap.
    """
    history, matches = get_history_with_matches(history_id)
    if not history or not matches:
        return None

    keywords = history.get("keywords") or []
    if not keywords:
        return None

    matrix = build_stage2_matrix(keywords, matches)
    if matrix is None:
        return None

    fig = build_heatmap_figure(matrix)
    return fig
