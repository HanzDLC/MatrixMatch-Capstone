from typing import Dict, Optional, Tuple

from matrixmatch_app.repositories import users


def authenticate_user(email: str, password: str) -> Tuple[Optional[Dict], Optional[Tuple[str, str]]]:
    email = (email or "").strip()
    password = (password or "").strip()

    if not email or not password:
        return None, ("Please fill in all fields.", "danger")

    user = users.get_user_by_credentials(email, password)
    if not user:
        return None, ("Invalid email or password.", "danger")

    return user, None


def register_user(
    first_name: str,
    last_name: str,
    email: str,
    password: str,
) -> Tuple[bool, Tuple[str, str]]:
    first_name = (first_name or "").strip()
    last_name = (last_name or "").strip()
    email = (email or "").strip()
    password = (password or "").strip()

    if not (first_name and last_name and email and password):
        return False, ("Please fill in all fields.", "danger")

    if users.get_user_by_email(email):
        return False, ("Email already registered.", "warning")

    created_id = users.create_researcher(
        first_name=first_name,
        last_name=last_name,
        email=email,
        password=password,
    )
    if not created_id:
        return False, ("Unable to create account. Please try again.", "danger")

    return True, ("Account created! You can now log in.", "success")
