from typing import Optional, Tuple

from matrixmatch_app.repositories import history, users


def list_researchers():
    return users.list_researchers()


def delete_researcher(researcher_id: int) -> bool:
    return users.delete_researcher_with_history(researcher_id)


def get_researcher_history(researcher_id: int):
    researcher = users.get_researcher_by_id(researcher_id)
    if not researcher:
        return None, []
    return researcher, history.list_history_for_user(researcher_id)


def validate_password_reset(
    new_password: str,
    confirm_password: str,
) -> Optional[Tuple[str, str]]:
    new_password = (new_password or "").strip()
    confirm_password = (confirm_password or "").strip()

    if not new_password or not confirm_password:
        return ("Please fill in both password fields.", "danger")
    if new_password != confirm_password:
        return ("Passwords do not match.", "danger")
    if len(new_password) < 6:
        return ("Password must be at least 6 characters long.", "warning")
    return None


def reset_researcher_password(researcher_id: int, new_password: str) -> None:
    users.update_user_password(researcher_id, new_password.strip())
