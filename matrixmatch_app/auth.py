from functools import wraps

from flask import flash, redirect, session, url_for


def get_current_user():
    """Return a dict-like object for the current logged-in user (from session)."""
    if session.get("is_guest"):
        return {
            "id": None,
            "first_name": "Guest",
            "last_name": "",
            "role": "Guest",
            "email": "",
        }
    if "user_id" not in session:
        return None
        
    from matrixmatch_app.repositories.users import get_user_by_id
    db_user = get_user_by_id(session["user_id"])
    
    if db_user:
        # Sync session with latest DB values so changes reflect across devices immediately
        session["first_name"] = db_user["first_name"]
        session["last_name"] = db_user["last_name"]
        session["profile_pic"] = db_user.get("profile_pic")
        
        return {
            "id": db_user["researcher_id"],
            "first_name": db_user["first_name"],
            "last_name": db_user["last_name"],
            "role": db_user["role"],
            "email": db_user["email"],
            "profile_pic": db_user.get("profile_pic"),
        }
        
    return None


def login_required(view_func):
    """Decorator to require login for protected views. Guests are allowed."""

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session and not session.get("is_guest"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped


def role_required(*allowed_roles):
    """Decorator to require one of the allowed roles for protected views."""

    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if "user_id" not in session and not session.get("is_guest"):
                flash("Please log in to continue.", "warning")
                return redirect(url_for("login"))

            current_role = session.get("role", "Guest") if not session.get("is_guest") else "Guest"
            if current_role not in allowed_roles:
                flash(f"{'/'.join(allowed_roles)} access only.", "danger")
                return redirect(url_for("dashboard"))

            return view_func(*args, **kwargs)

        return wrapped

    return decorator

