from matrixmatch_app.db import db_cursor

def add_log(document_id: int, document_title: str, action: str, modified_by: str, cursor=None):
    if cursor is None:
        with db_cursor(commit=True) as cursor:
            add_log(document_id, document_title, action, modified_by, cursor=cursor)
            return

    cursor.execute(
        """
        INSERT INTO matrixmatch.documents_log (log_id, document_id, document_title, action, modified_by)
        VALUES (nextval('matrixmatch.documents_log_log_id_seq'::regclass), %s, %s, %s, %s)
        """,
        (document_id, document_title, action, modified_by)
    )

def get_all_logs():
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT log_id, document_id, document_title, action, modified_by, timestamp
            FROM matrixmatch.documents_log
            ORDER BY timestamp DESC
            """
        )
        return cursor.fetchall()
