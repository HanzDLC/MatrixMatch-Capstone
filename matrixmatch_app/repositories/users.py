from typing import Optional

from matrixmatch_app.db import db_cursor


def get_user_by_credentials(email: str, password: str):
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT researcher_id, first_name, last_name, email, role
            FROM "user"
            WHERE email = %s AND password = %s
            """,
            (email, password),
        )
        return cursor.fetchone()


def get_user_by_email(email: str):
    with db_cursor() as cursor:
        cursor.execute(
            'SELECT researcher_id, first_name, last_name, email, role FROM "user" WHERE email = %s',
            (email,),
        )
        return cursor.fetchone()


def create_researcher(
    first_name: str,
    last_name: str,
    email: str,
    password: str,
) -> Optional[int]:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            INSERT INTO "user" (first_name, last_name, email, password, role)
            VALUES (%s, %s, %s, %s, 'Researcher')
            RETURNING researcher_id
            """,
            (first_name, last_name, email, password),
        )
        created = cursor.fetchone()
        return created["researcher_id"] if created else None


def get_user_by_id(user_id: int):
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT researcher_id, first_name, last_name, email, role
            FROM "user"
            WHERE researcher_id = %s
            """,
            (user_id,),
        )
        return cursor.fetchone()


def get_researcher_by_id(researcher_id: int):
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT researcher_id, first_name, last_name, email
            FROM "user"
            WHERE researcher_id = %s AND role = 'Researcher'
            """,
            (researcher_id,),
        )
        return cursor.fetchone()


def list_researchers():
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT researcher_id, first_name, last_name, email, registered_date
            FROM "user"
            WHERE role = 'Researcher'
            ORDER BY registered_date DESC
            """
        )
        return cursor.fetchall()


def delete_researcher_with_history(researcher_id: int) -> bool:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            "DELETE FROM comparison_history WHERE researcher_id = %s",
            (researcher_id,),
        )
        cursor.execute(
            'DELETE FROM "user" WHERE researcher_id = %s AND role = \'Researcher\'',
            (researcher_id,),
        )
        return cursor.rowcount > 0


def update_user_password(researcher_id: int, new_password: str) -> None:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE "user"
            SET password = %s
            WHERE researcher_id = %s
            """,
            (new_password, researcher_id),
        )
