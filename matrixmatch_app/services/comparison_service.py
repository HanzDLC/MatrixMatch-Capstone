import base64
import importlib
import io
import logging
from typing import Dict, List, Optional, Tuple

from matrixmatch_app.parsers import parse_keywords

logger = logging.getLogger(__name__)


def _get_matcher_module():
    try:
        return importlib.import_module("matcher")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Comparison dependencies are not installed. Install requirements.txt to enable comparison features."
        ) from exc


def parse_threshold(threshold_str: str, default_pct: float = 60.0) -> float:
    try:
        threshold_pct = float((threshold_str or "").strip())
    except ValueError:
        threshold_pct = default_pct

    threshold_pct = min(max(threshold_pct, 0.0), 100.0)
    return threshold_pct / 100.0


def run_new_comparison(
    researcher_id: int,
    raw_keywords: str,
    user_abstract: str,
    program_filter: str,
    threshold_str: str,
) -> Tuple[Optional[int], Optional[List[Dict]], Optional[Tuple[str, str]]]:
    raw_keywords = (raw_keywords or "").strip()
    user_abstract = (user_abstract or "").strip()
    program_filter = (program_filter or "ALL").strip() or "ALL"

    if not raw_keywords or not user_abstract:
        return None, None, ("Please enter both keywords and an abstract.", "danger")

    keywords = parse_keywords(raw_keywords)
    if len(keywords) < 5:
        return None, None, ("Please enter at least 5 keywords.", "danger")

    similarity_threshold = parse_threshold(threshold_str)

    try:
        matcher = _get_matcher_module()
    except RuntimeError as exc:
        logger.exception("Comparison requested without matcher dependencies.")
        return None, None, (str(exc), "danger")

    history_id, matches = matcher.run_stage1(
        researcher_id=researcher_id,
        keywords=keywords,
        user_abstract=user_abstract,
        academic_program_filter=program_filter,
        similarity_threshold=similarity_threshold,
    )

    if history_id is None:
        return None, None, ("No documents found for the selected program.", "warning")

    return history_id, matches, None


def build_history_heatmap_data_uri(keywords: List[str], matches: List[Dict]) -> Optional[str]:
    if not keywords or not matches:
        return None

    try:
        matcher = _get_matcher_module()
    except RuntimeError:
        logger.exception("Heatmap requested without matcher dependencies.")
        return None

    try:
        matrix = matcher.build_stage2_matrix(keywords, matches)
        if matrix is None:
            return None

        fig = matcher.build_heatmap_figure(matrix)
    except Exception:
        logger.exception("Failed to build heatmap image for history detail.")
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    image_b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{image_b64}"


def build_history_heatmap_table(keywords: List[str], matches: List[Dict]):
    try:
        matcher = _get_matcher_module()
    except RuntimeError:
        logger.exception("Heatmap table requested without matcher dependencies.")
        return None

    try:
        matrix = matcher.build_stage2_matrix(keywords, matches)
    except Exception:
        logger.exception("Failed to build heatmap table for history detail.")
        return None

    if matrix is None or matrix.empty:
        return None

    col_labels = list(matrix.columns)
    row_labels = list(matrix.index)
    values = matrix.values.tolist()

    table_rows = []
    for row_index, keyword in enumerate(row_labels):
        table_rows.append(
            {
                "keyword": keyword,
                "cells": [
                    {"col_label": col_labels[col_index], "value": values[row_index][col_index]}
                    for col_index in range(len(col_labels))
                ],
            }
        )

    return {
        "col_labels": col_labels,
        "table_rows": table_rows,
        "min_val": float(matrix.values.min()),
        "max_val": float(matrix.values.max()),
    }
