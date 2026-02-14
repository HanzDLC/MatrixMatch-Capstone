from matrixmatch_app.db import db_cursor


def get_admin_stats():
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN role = 'Researcher' THEN 1 ELSE 0 END), 0) AS total_researchers,
                COALESCE(SUM(CASE WHEN role = 'Admin' THEN 1 ELSE 0 END), 0) AS total_admins,
                (SELECT COUNT(*) FROM comparison_history) AS total_comparisons,
                (
                    SELECT COUNT(*)
                    FROM comparison_history
                    WHERE created_at >= NOW() - INTERVAL '7 days'
                ) AS last_7_days_runs
            FROM "user"
            """
        )
        return cursor.fetchone()


def list_recent_history(limit: int = 10):
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT ch.history_id,
                   CONCAT(u.first_name, ' ', u.last_name) AS researcher_name,
                   ch.academic_program_filter,
                   ch.similarity_threshold,
                   ch.created_at
            FROM comparison_history ch
            JOIN "user" u ON ch.researcher_id = u.researcher_id
            ORDER BY ch.created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return cursor.fetchall()


def list_recent_history_for_user(researcher_id: int, limit: int = 5):
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT history_id,
                   keywords,
                   academic_program_filter,
                   similarity_threshold,
                   created_at
            FROM comparison_history
            WHERE researcher_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (researcher_id, limit),
        )
        return cursor.fetchall()


def list_history_for_user(researcher_id: int):
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT history_id,
                   keywords,
                   academic_program_filter,
                   similarity_threshold,
                   created_at
            FROM comparison_history
            WHERE researcher_id = %s
            ORDER BY created_at DESC
            """,
            (researcher_id,),
        )
        return cursor.fetchall()
