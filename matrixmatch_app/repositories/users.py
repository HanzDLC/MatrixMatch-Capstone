from typing import Optional

from matrixmatch_app.db import db_cursor


def get_user_by_credentials(email: str, password: str):
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT researcher_id, first_name, last_name, email, role, profile_pic
            FROM matrixmatch."user"
            WHERE email = %s AND password = %s
            """,
            (email, password),
        )
        return cursor.fetchone()


def get_user_by_email(email: str):
    with db_cursor() as cursor:
        cursor.execute(
            'SELECT researcher_id, first_name, last_name, email, role, profile_pic FROM matrixmatch."user" WHERE email = %s',
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
            INSERT INTO matrixmatch."user" (first_name, last_name, email, password, role)
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
            SELECT researcher_id, first_name, last_name, email, role, profile_pic
            FROM matrixmatch."user"
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
            FROM matrixmatch."user"
            WHERE researcher_id = %s AND role = 'Researcher'
            """,
            (researcher_id,),
        )
        return cursor.fetchone()


def list_researchers():
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT researcher_id, first_name, last_name, email, role, registered_date
            FROM matrixmatch."user"
            ORDER BY registered_date DESC
            """
        )
        return cursor.fetchall()


def delete_researcher_with_history(researcher_id: int) -> bool:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            "DELETE FROM matrixmatch.comparison_history WHERE researcher_id = %s",
            (researcher_id,),
        )
        cursor.execute(
            'DELETE FROM matrixmatch."user" WHERE researcher_id = %s AND role = \'Researcher\'',
            (researcher_id,),
        )
        return cursor.rowcount > 0


def update_user_password(researcher_id: int, new_password: str) -> None:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE matrixmatch."user"
            SET password = %s
            WHERE researcher_id = %s
            """,
            (new_password, researcher_id),
        )

def update_user_profile(user_id: int, first_name: str, last_name: str) -> None:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE matrixmatch."user"
            SET first_name = %s, last_name = %s
            WHERE researcher_id = %s
            """,
            (first_name, last_name, user_id),
        )

def promote_researcher_to_admin(researcher_id: int) -> bool:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE matrixmatch."user"
            SET role = 'Admin'
            WHERE researcher_id = %s AND role = 'Researcher'
            """,
            (researcher_id,),
        )
        return cursor.rowcount > 0


def demote_admin_to_researcher(researcher_id: int, protected_email: str) -> bool:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE matrixmatch."user"
            SET role = 'Researcher'
            WHERE researcher_id = %s
              AND role = 'Admin'
              AND LOWER(email) <> LOWER(%s)
            """,
            (researcher_id, protected_email),
        )
        return cursor.rowcount > 0


def update_profile_pic(user_id: int, filename: str) -> None:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE matrixmatch."user"
            SET profile_pic = %s
            WHERE researcher_id = %s
            """,
            (filename, user_id),
        )

def update_last_seen(user_id: int) -> None:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE matrixmatch."user"
            SET last_seen = NOW()
            WHERE researcher_id = %s
            """,
            (user_id,),
        )

def get_online_users(minutes: int = 5):
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT researcher_id, first_name, last_name, role, profile_pic, last_seen
            FROM matrixmatch."user"
            WHERE last_seen >= NOW() - INTERVAL '1 minute' * %s
            ORDER BY last_seen DESC
            """,
            (minutes,),
        )
        return cursor.fetchall()
