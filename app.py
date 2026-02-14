from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_file
)
import psycopg2
from psycopg2.extras import RealDictCursor
import base64
import io
import matcher
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os

# -----------------------------
# Flask app setup
# -----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey_change_me")  # TODO: change in production

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "dbname": os.getenv("DB_NAME", "matrixmatch"),
    "options": os.getenv("DB_OPTIONS", "-c search_path=matrixmatch,public")
}


def get_db_connection():
    """Create a new DB connection."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


# -----------------------------
# Helpers
# -----------------------------
def get_current_user():
    """Return a dict-like object for the current logged-in user (from session)."""
    if "user_id" not in session:
        return None
    return {
        "id": session.get("user_id"),
        "first_name": session.get("first_name"),
        "last_name": session.get("last_name"),
        "role": session.get("role"),
        "email": session.get("email"),
    }


def login_required(view_func):
    """Simple decorator to require login for certain routes."""
    from functools import wraps

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped


# -----------------------------
# Routes
# -----------------------------

# Landing page
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Auth: Login & Register
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    # GET -> show login form
    if request.method == "GET":
        return render_template("login.html")

    # POST -> process login
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()

    if not email or not password:
        flash("Please fill in all fields.", "danger")
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # v1 schema: single `user` table with role ENUM('Admin','Researcher')
        cursor.execute(
            """
            SELECT researcher_id, first_name, last_name, email, role
            FROM "user"
            WHERE email = %s AND password = %s
            """,
            (email, password)
        )
        user = cursor.fetchone()
    finally:
        cursor.close()
        conn.close()

    if not user:
        flash("Invalid email or password.", "danger")
        return redirect(url_for("login"))

    # Save to session
    session["user_id"] = user["researcher_id"]
    session["first_name"] = user["first_name"]
    session["last_name"] = user["last_name"]
    session["role"] = user["role"]      # 'Admin' or 'Researcher'
    session["email"] = user["email"]

    flash(f"Welcome back, {user['first_name']}!", "success")
    return redirect(url_for("dashboard"))


@app.route("/register", methods=["GET", "POST"])
def register():
    # Show the registration form
    if request.method == "GET":
        return render_template("register.html")

    # Handle form submission
    first_name = request.form.get("first_name", "").strip()
    last_name = request.form.get("last_name", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()

    if not (first_name and last_name and email and password):
        flash("Please fill in all fields.", "danger")
        return redirect(url_for("register"))

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Check if email already exists
        cursor.execute('SELECT researcher_id FROM "user" WHERE email = %s', (email,))
        existing = cursor.fetchone()
        if existing:
            flash("Email already registered.", "warning")
            return redirect(url_for("register"))

        # Insert new researcher (role = 'Researcher')
        cursor.execute(
            """
            INSERT INTO "user" (first_name, last_name, email, password, role)
            VALUES (%s, %s, %s, %s, 'Researcher')
            """,
            (first_name, last_name, email, password)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    flash("Account created! You can now log in.", "success")
    return redirect(url_for("login"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# -----------------------------
# Dashboard routing
# -----------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    """
    Generic endpoint used in base.html via url_for('dashboard').
    Redirects to the appropriate dashboard based on role.
    """
    role = session.get("role", "")
    if role == "Admin":
        return redirect(url_for("admin_dashboard"))
    elif role == "Researcher":
        return redirect(url_for("researcher_dashboard"))
    else:
        # fallback to login if something weird happens
        session.clear()
        return redirect(url_for("login"))


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    if session.get("role") != "Admin":
        flash("Admin access only.", "danger")
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Fetch admin user info
        cursor.execute(
            "SELECT researcher_id, first_name, last_name, email, role "
            'FROM "user" WHERE researcher_id = %s',
            (session["user_id"],)
        )
        user = cursor.fetchone()

        # ----- Stats -----
        cursor.execute(
            'SELECT COUNT(*) AS total_researchers FROM "user" WHERE role=\'Researcher\''
        )
        total_researchers = cursor.fetchone()["total_researchers"]

        cursor.execute(
            'SELECT COUNT(*) AS total_admins FROM "user" WHERE role=\'Admin\''
        )
        total_admins = cursor.fetchone()["total_admins"]

        cursor.execute(
            "SELECT COUNT(*) AS total_comparisons FROM comparison_history"
        )
        total_comparisons = cursor.fetchone()["total_comparisons"]

        stats = {
            "total_researchers": total_researchers,
            "total_admins": total_admins,
            "total_comparisons": total_comparisons
        }

        # ----- Load ALL recent comparison history with full researcher name -----
        cursor.execute(
            """
            SELECT ch.history_id,
                   CONCAT(u.first_name, ' ', u.last_name) AS researcher_name,
                   ch.academic_program_filter,
                   ch.similarity_threshold,
                   ch.created_at
            FROM comparison_history ch
            JOIN user u ON ch.researcher_id = u.researcher_id
            ORDER BY ch.created_at DESC
            LIMIT 10
            """
        )
        recent_history = cursor.fetchall()

    finally:
        cursor.close()
        conn.close()

    return render_template(
        "dashboard_admin.html",
        user=user,
        stats=stats,
        recent_history=recent_history
    )



@app.route("/researcher/dashboard")
@login_required
def researcher_dashboard():
    """
    Researcher dashboard showing their own recent comparison history.
    """
    user = get_current_user()
    if not user or user["role"] != "Researcher":
        flash("Researcher access only.", "danger")
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get recent comparison history for this researcher
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
            LIMIT 5
            """,
            (user["id"],)
        )
        recent_history = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    return render_template(
        "dashboard_researcher.html",
        user=user,
        recent_history=recent_history
    )


# -----------------------------
# Comparison (Stage 1 + 2 placeholder)
# -----------------------------
@app.route("/comparison/new", methods=["GET", "POST"])
@login_required
def comparison_new():
    """
    Stage 1 entry point.
    - GET: show the comparison form.
    - POST: run Stage 1, save to comparison_history, redirect to history_detail.
    """
    import json  # or put this at the top of the file

    user = get_current_user()

    if request.method == "GET":
        return render_template("comparison_new.html", user=user)

    # -----------------------------
    # POST form submission
    # -----------------------------
    raw_keywords = request.form.get("keywords", "").strip()
    user_abstract = request.form.get("abstract", "").strip()
    program_filter = request.form.get("program_filter", "ALL").strip() or "ALL"
    threshold_str = request.form.get("threshold", "60").strip()

    # basic validations
    if not raw_keywords or not user_abstract:
        flash("Please enter both keywords and an abstract.", "danger")
        return redirect(url_for("comparison_new"))

    # -----------------------------
    # Parse threshold (percent -> 0-1)
    # -----------------------------
    try:
        threshold_pct = float(threshold_str)
    except ValueError:
        threshold_pct = 60.0
    similarity_threshold = threshold_pct / 100.0

    # -----------------------------
    # Parse keywords (handles BOTH JSON array and comma-separated)
    # -----------------------------
    keywords = []

    if raw_keywords:
        # Case 1: looks like JSON array -> try json.loads
        if raw_keywords.lstrip().startswith("["):
            try:
                parsed = json.loads(raw_keywords)
                if isinstance(parsed, list):
                    keywords = [str(k).strip() for k in parsed if str(k).strip()]
                else:
                    # Fallback to comma-split if it's not actually a list
                    keywords = [
                        part.strip()
                        for part in str(parsed).split(",")
                        if part.strip()
                    ]
            except json.JSONDecodeError:
                # Fallback: treat as plain comma-separated
                keywords = [
                    part.strip()
                    for part in raw_keywords.split(",")
                    if part.strip()
                ]
        else:
            # Case 2: plain comma-separated string
            keywords = [
                part.strip()
                for part in raw_keywords.split(",")
                if part.strip()
            ]

    if len(keywords) < 5:
        flash("Please enter at least 5 keywords.", "danger")
        return redirect(url_for("comparison_new"))

    # -----------------------------
    # Stage 1 call
    # matcher.run_stage1 is responsible for:
    # - computing similarities
    # - saving to comparison_history
    # - returning (history_id, matches)
    # -----------------------------
    history_id, matches = matcher.run_stage1(
        researcher_id=user["id"],
        keywords=keywords,                 # clean Python list
        user_abstract=user_abstract,
        academic_program_filter=program_filter,
        similarity_threshold=similarity_threshold,
    )

    if history_id is None:
        flash("No documents found for the selected program.", "warning")
        return redirect(url_for("comparison_new"))

    # Redirect to history detail for this run (Stage 1 results)
    flash("Stage 1 comparison completed.", "success")
    return redirect(url_for("history_detail", history_id=history_id))


# -----------------------------
# History (list + detail placeholders)
# -----------------------------
@app.route("/history")
@login_required
def history():
    """
    History list view.

    For BOTH Admin and Researcher:
    - This page shows ONLY the currently logged-in user's own history
      (comparison_history rows where researcher_id = session user_id).

    Admins still see ALL users' history on the admin dashboard overview.
    """
    user = get_current_user()

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT
                history_id,
                keywords,
                academic_program_filter,
                similarity_threshold,
                created_at
            FROM comparison_history
            WHERE researcher_id = %s
            ORDER BY created_at DESC
            """,
            (user["id"],)
        )
        history_rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    return render_template(
        "history.html",
        user=user,
        history_rows=history_rows
    )



@app.route("/history/<int:history_id>")
@login_required
def history_detail(history_id):
    """
    Single history detail page (Stage 1 list + Stage 2 heatmap).
    - Reloads keywords + Stage 1 matches from DB via matcher.get_history_with_matches().
    - Runs Stage 2 (matrix + heatmap) immediately when this page is loaded.
    - Embeds the heatmap as a base64 <img> in the template.
    """
    user = get_current_user()

    # Load history row + Stage 1 matches from DB
    history, matches = matcher.get_history_with_matches(history_id)
    if not history:
        flash("History entry not found.", "warning")
        return redirect(url_for("history"))

    # Permission: researchers can only see their own; admins can see all
    if user["role"] == "Researcher" and history["researcher_id"] != user["id"]:
        flash("You are not allowed to view that history entry.", "danger")
        return redirect(url_for("history"))

    # -----------------------------
    # Parse keywords from history["keywords"]
    # -----------------------------
    raw_keywords = history.get("keywords")
    keywords = []

    if isinstance(raw_keywords, str):
        # Try JSON first (e.g. '["AI", "Vector DB", ...]')
        try:
            parsed = json.loads(raw_keywords)
            if isinstance(parsed, list):
                keywords = [str(k).strip() for k in parsed if str(k).strip()]
            else:
                # Fallback: comma-separated
                keywords = [
                    part.strip()
                    for part in str(parsed).split(",")
                    if part.strip()
                ]
        except json.JSONDecodeError:
            # Fallback: plain comma-separated string
            keywords = [
                part.strip()
                for part in raw_keywords.split(",")
                if part.strip()
            ]
    elif isinstance(raw_keywords, list):
        keywords = [str(k).strip() for k in raw_keywords if str(k).strip()]

    # -----------------------------
    # Stage 2: build matrix + figure
    # -----------------------------
    heatmap_data_uri = None
    if keywords and matches:
        matrix = matcher.build_stage2_matrix(keywords, matches)
        if matrix is not None:
            fig = matcher.build_heatmap_figure(matrix)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)

            img_b64 = base64.b64encode(buf.read()).decode("ascii")
            heatmap_data_uri = f"data:image/png;base64,{img_b64}"

    return render_template(
        "history_detail.html",
        user=user,
        history=history,
        matches=matches,           # Stage 1 results (with .program populated)
        keywords=keywords,         # for the keyword chips
        heatmap_data_uri=heatmap_data_uri,  # Stage 2 heatmap (if available)

    )



# -----------------------------
# Manage Researchers (Admin-only)
# -----------------------------
@app.route("/admin/researchers")
@login_required
def manage_researchers():
    """
    Admin view to manage researchers (matches url_for('manage_researchers')).
    Now wired to the `user` table with role='Researcher'.
    """
    user = get_current_user()
    if not user or user["role"] != "Admin":
        flash("Admin access only.", "danger")
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT researcher_id, first_name, last_name, email, registered_date
            FROM "user"
            WHERE role = 'Researcher'
            ORDER BY registered_date DESC
            """
        )
        researcher_rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    return render_template(
        "manage_researchers.html",
        user=user,
        researchers=researcher_rows
    )

# -----------------------------
# Admin: researcher actions
# -----------------------------
# @app.route("/admin/researchers/<int:researcher_id>/reset-password", methods=["POST"])
# @login_required
# def admin_reset_password(researcher_id):
#     """Admin: reset a researcher's password to a default value."""
#     user = get_current_user()
#     if user["role"] != "Admin":
#         flash("Admin access only.", "danger")
#         return redirect(url_for("dashboard"))
#
#     # Example: simple default password (you can improve this later)
#     new_password = "matrix123"  # TODO: generate a random secure password in future
#
#     conn = get_db_connection()
#     cursor = conn.cursor()
#
#     try:
#         cursor.execute(
#             """
#             UPDATE "user"
#             SET password = %s
#             WHERE researcher_id = %s AND role = 'Researcher'
#             """,
#             (new_password, researcher_id)
#         )
#         conn.commit()
#     finally:
#         cursor.close()
#         conn.close()
#
#     flash(f"Password has been reset to '{new_password}' for researcher ID {researcher_id}.", "success")
#     return redirect(url_for("manage_researchers"))


@app.route("/admin/researchers/<int:researcher_id>/delete", methods=["POST"])
@login_required
def admin_delete_researcher(researcher_id):
    """Admin: delete a researcher account (and optionally their history)."""
    user = get_current_user()
    if user["role"] != "Admin":
        flash("Admin access only.", "danger")
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Optional: also delete comparison_history rows
        cursor.execute(
            "DELETE FROM comparison_history WHERE researcher_id = %s",
            (researcher_id,)
        )

        cursor.execute(
            'DELETE FROM "user" WHERE researcher_id = %s AND role = \'Researcher\'',
            (researcher_id,)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    flash(f"Researcher ID {researcher_id} has been deleted.", "info")
    return redirect(url_for("manage_researchers"))


@app.route("/admin/researchers/<int:researcher_id>/history")
@login_required
def admin_view_history(researcher_id):
    """Admin: view history for a specific researcher."""
    user = get_current_user()
    if user["role"] != "Admin":
        flash("Admin access only.", "danger")
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get researcher info
        cursor.execute(
            """
            SELECT researcher_id, first_name, last_name, email
            FROM "user"
            WHERE researcher_id = %s AND role = 'Researcher'
            """,
            (researcher_id,)
        )
        researcher = cursor.fetchone()

        if not researcher:
            flash("Researcher not found.", "warning")
            return redirect(url_for("manage_researchers"))

        # Get their history
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
            (researcher_id,)
        )
        history_rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    # Reuse the history template (you can also create a separate admin-specific one later)
    return render_template(
        "history.html",
        user=user,                    # current admin
        history_rows=history_rows,
        selected_researcher=researcher
    )

@app.route("/admin/researchers/<int:researcher_id>/reset", methods=["GET", "POST"])
@login_required
def admin_reset_password(researcher_id):
    # Only admins can access this
    if session.get("role") != "Admin":
        flash("Admin access only.", "danger")
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get the researcher record
        cursor.execute(
            """
            SELECT researcher_id, first_name, last_name, email
            FROM "user"
            WHERE researcher_id = %s AND role = 'Researcher'
            """,
            (researcher_id,)
        )
        researcher = cursor.fetchone()

        if not researcher:
            flash("Researcher not found.", "danger")
            return redirect(url_for("manage_researchers"))

        # GET -> show the reset form
        if request.method == "GET":
            return render_template(
                "admin_reset_password.html",
                user=get_current_user(),
                researcher=researcher
            )

        # POST -> update password
        new_password = request.form.get("new_password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not new_password or not confirm_password:
            flash("Please fill in both password fields.", "danger")
            return redirect(url_for("admin_reset_password", researcher_id=researcher_id))

        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("admin_reset_password", researcher_id=researcher_id))

        if len(new_password) < 6:
            flash("Password must be at least 6 characters long.", "warning")
            return redirect(url_for("admin_reset_password", researcher_id=researcher_id))

        # Update the password in the DB (plain text for now, same as your existing login)
        cursor.execute(
            """
            UPDATE "user"
            SET password = %s
            WHERE researcher_id = %s
            """,
            (new_password, researcher_id)
        )
        conn.commit()

        flash(f"Password updated for {researcher['first_name']} {researcher['last_name']}.", "success")
        return redirect(url_for("manage_researchers"))

    finally:
        cursor.close()
        conn.close()

@app.route("/history/<int:history_id>/heatmap")
@login_required
def history_heatmap(history_id):
    """
    Show Stage 2 as a full-page HTML table with a color gradient.
    Opens in a new tab from the History Detail page.
    """
    user = get_current_user()

    # Load history + Stage 1 matches
    history, matches = matcher.get_history_with_matches(history_id)
    if not history:
        flash("History entry not found.", "danger")
        return redirect(url_for("history"))

    # Permission check: researchers see only their own
    if user["role"] == "Researcher" and history["researcher_id"] != user["id"]:
        flash("You are not allowed to view that history entry.", "danger")
        return redirect(url_for("history"))

    # ----- Parse keywords from history["keywords"] -----
    raw_keywords = history.get("keywords") or ""
    keywords = []

    if isinstance(raw_keywords, str):
        try:
            parsed = json.loads(raw_keywords)
            if isinstance(parsed, list):
                keywords = [str(k).strip() for k in parsed if str(k).strip()]
            else:
                keywords = [
                    part.strip()
                    for part in str(parsed).split(",")
                    if part.strip()
                ]
        except json.JSONDecodeError:
            keywords = [
                part.strip()
                for part in raw_keywords.split(",")
                if part.strip()
            ]
    elif isinstance(raw_keywords, list):
        keywords = [str(k).strip() for k in raw_keywords if str(k).strip()]

    if not keywords or not matches:
        flash("Not enough data to build a heatmap for this entry.", "warning")
        return redirect(url_for("history_detail", history_id=history_id))

    # ----- Build Stage 2 matrix (keyword x document) -----
    matrix = matcher.build_stage2_matrix(keywords, matches)
    if matrix is None or matrix.empty:
        flash("Unable to build heatmap matrix for this entry.", "warning")
        return redirect(url_for("history_detail", history_id=history_id))

    col_labels = list(matrix.columns)         # documents
    row_labels = list(matrix.index)           # keywords
    values = matrix.values.tolist()           # 2D list of floats (0-1)

    min_val = float(matrix.values.min())
    max_val = float(matrix.values.max())

    # Build a friendlier structure for Jinja
    table_rows = []
    for i, kw in enumerate(row_labels):
        row = {
            "keyword": kw,
            "cells": [
                {"col_label": col_labels[j], "value": values[i][j]}
                for j in range(len(col_labels))
            ],
        }
        table_rows.append(row)

    return render_template(
        "history_heatmap_table.html",
        user=user,
        history=history,
        col_labels=col_labels,
        table_rows=table_rows,
        min_val=min_val,
        max_val=max_val,
    )




# -----------------------------
# Main entrypoint
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

