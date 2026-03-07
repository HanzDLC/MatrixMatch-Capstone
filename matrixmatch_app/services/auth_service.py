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

def update_profile(
    user_id: int,
    first_name: str,
    last_name: str,
    old_password: str,
    new_password: str,
    confirm_password: str
) -> Tuple[bool, Tuple[str, str]]:
    first_name = (first_name or "").strip()
    last_name = (last_name or "").strip()
    old_password = (old_password or "").strip()
    new_password = (new_password or "").strip()
    confirm_password = (confirm_password or "").strip()

    if not first_name or not last_name:
        return False, ("First name and last name are required.", "danger")

    if new_password:
        if not old_password:
            return False, ("Current password is required to set a new password.", "danger")
        
        user_info = users.get_user_by_id(user_id)
        if not user_info:
            return False, ("User not found.", "danger")
            
        authenticated_user = users.get_user_by_credentials(user_info["email"], old_password)
        if not authenticated_user:
            return False, ("Incorrect current password.", "danger")

        if new_password != confirm_password:
            return False, ("Passwords do not match.", "danger")
        if len(new_password) < 6:
            return False, ("Password must be at least 6 characters.", "warning")
        users.update_user_password(user_id, new_password)

    users.update_user_profile(user_id, first_name, last_name)
    return True, ("Profile updated successfully.", "success")
