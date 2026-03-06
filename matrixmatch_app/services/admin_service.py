from typing import Optional, Tuple

from matrixmatch_app.repositories import history, users

PROTECTED_ADMIN_EMAIL = "admin@gmail.com"


def list_researchers():
    return users.list_researchers()


def delete_researcher(researcher_id: int) -> bool:
    return users.delete_researcher_with_history(researcher_id)


def get_researcher_history(researcher_id: int):
    researcher = users.get_user_by_id(researcher_id)
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


def promote_researcher(researcher_id: int) -> bool:
    return users.promote_researcher_to_admin(researcher_id)


def demote_admin(researcher_id: int) -> Tuple[bool, Tuple[str, str]]:
    account = users.get_user_by_id(researcher_id)
    if not account:
        return False, ("User not found.", "warning")

    if account.get("role") != "Admin":
        return False, ("Only admin accounts can be demoted.", "warning")

    email = (account.get("email") or "").strip().lower()
    if email == PROTECTED_ADMIN_EMAIL:
        return False, (f"{PROTECTED_ADMIN_EMAIL} cannot be demoted.", "warning")

    demoted = users.demote_admin_to_researcher(researcher_id, PROTECTED_ADMIN_EMAIL)
    if not demoted:
        return False, ("Admin could not be demoted.", "warning")

    full_name = f"{account.get('first_name', '')} {account.get('last_name', '')}".strip()
    return True, (f"{full_name or 'Admin'} has been demoted to Researcher.", "success")
