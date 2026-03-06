from typing import List, Optional, Dict

from matrixmatch_app.db import db_cursor


def list_all_documents() -> List[Dict]:
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT document_id, title, academic_program, abstract
            FROM matrixmatch.documents
            ORDER BY document_id DESC
            """
        )
        return cursor.fetchall()


def get_document_by_id(document_id: int) -> Optional[Dict]:
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT document_id, title, academic_program, abstract
            FROM matrixmatch.documents
            WHERE document_id = %s
            """,
            (document_id,)
        )
        return cursor.fetchone()


def add_document(title: str, academic_program: str, abstract: str) -> Optional[int]:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            INSERT INTO matrixmatch.documents (title, academic_program, abstract)
            VALUES (%s, %s, %s)
            RETURNING document_id
            """,
            (title, academic_program, abstract)
        )
        created = cursor.fetchone()
        return created["document_id"] if created else None


def update_document(document_id: int, title: str, academic_program: str, abstract: str) -> bool:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE matrixmatch.documents
            SET title = %s,
                academic_program = %s,
                abstract = %s
            WHERE document_id = %s
            """,
            (title, academic_program, abstract, document_id)
        )
        return cursor.rowcount > 0


def delete_document(document_id: int) -> bool:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            DELETE FROM matrixmatch.documents
            WHERE document_id = %s
            """,
            (document_id,)
        )
        return cursor.rowcount > 0

def check_duplicate_document(title: str, abstract: str, exclude_id: Optional[int] = None) -> bool:
    with db_cursor() as cursor:
        query = """
            SELECT 1 FROM matrixmatch.documents
            WHERE (title = %s OR abstract = %s)
        """
        params = [title, abstract]
        if exclude_id is not None:
            query += " AND document_id != %s"
            params.append(exclude_id)
            
        cursor.execute(query, tuple(params))
        return cursor.fetchone() is not None

