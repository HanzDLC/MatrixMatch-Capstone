from flask import flash, redirect, render_template, request, session, url_for

import matcher

from matrixmatch_app.auth import get_current_user, login_required, role_required
from matrixmatch_app.parsers import parse_keywords
from matrixmatch_app.repositories import history as history_repo
from matrixmatch_app.repositories import users as users_repo
from matrixmatch_app.services import admin_service, auth_service, comparison_service, dashboard_service


def register_routes(app):
    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("login.html")

        user, error = auth_service.authenticate_user(
            email=request.form.get("email", ""),
            password=request.form.get("password", ""),
        )
        if error:
            flash(error[0], error[1])
            return redirect(url_for("login"))

        session["user_id"] = user["researcher_id"]
        session["first_name"] = user["first_name"]
        session["last_name"] = user["last_name"]
        session["role"] = user["role"]
        session["email"] = user["email"]

        flash(f"Welcome back, {user['first_name']}!", "success")
        return redirect(url_for("dashboard"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "GET":
            return render_template("register.html")

        created, outcome = auth_service.register_user(
            first_name=request.form.get("first_name", ""),
            last_name=request.form.get("last_name", ""),
            email=request.form.get("email", ""),
            password=request.form.get("password", ""),
        )
        flash(outcome[0], outcome[1])
        if not created:
            return redirect(url_for("register"))

        return redirect(url_for("login"))

    @app.route("/logout")
    def logout():
        session.clear()
        flash("You have been logged out.", "info")
        return redirect(url_for("login"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        role = session.get("role", "")
        if role == "Admin":
            return redirect(url_for("admin_dashboard"))
        if role == "Researcher":
            return redirect(url_for("researcher_dashboard"))

        session.clear()
        return redirect(url_for("login"))

    @app.route("/admin/dashboard")
    @role_required("Admin")
    def admin_dashboard():
        data = dashboard_service.get_admin_dashboard_data(session["user_id"])
        if not data:
            session.clear()
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))

        return render_template(
            "dashboard_admin.html",
            user=data["user"],
            stats=data["stats"],
            recent_history=data["recent_history"],
        )

    @app.route("/researcher/dashboard")
    @role_required("Researcher")
    def researcher_dashboard():
        user = get_current_user()
        data = dashboard_service.get_researcher_dashboard_data(user["id"])
        return render_template(
            "dashboard_researcher.html",
            user=user,
            recent_history=data["recent_history"],
        )

    @app.route("/comparison/new", methods=["GET", "POST"])
    @login_required
    def comparison_new():
        user = get_current_user()
        if request.method == "GET":
            return render_template("comparison_new.html", user=user)

        history_id, _matches, error = comparison_service.run_new_comparison(
            researcher_id=user["id"],
            raw_keywords=request.form.get("keywords", ""),
            user_abstract=request.form.get("abstract", ""),
            program_filter=request.form.get("program_filter", "ALL"),
            threshold_str=request.form.get("threshold", "60"),
        )
        if error:
            flash(error[0], error[1])
            return redirect(url_for("comparison_new"))

        flash("Stage 1 comparison completed.", "success")
        return redirect(url_for("history_detail", history_id=history_id))

    @app.route("/history")
    @login_required
    def history():
        user = get_current_user()
        history_rows = history_repo.list_history_for_user(user["id"])

        return render_template(
            "history.html",
            user=user,
            history_rows=history_rows,
        )

    @app.route("/history/<int:history_id>")
    @login_required
    def history_detail(history_id):
        user = get_current_user()
        history_entry, matches = matcher.get_history_with_matches(history_id)
        if not history_entry:
            flash("History entry not found.", "warning")
            return redirect(url_for("history"))

        if user["role"] == "Researcher" and history_entry["researcher_id"] != user["id"]:
            flash("You are not allowed to view that history entry.", "danger")
            return redirect(url_for("history"))

        keywords = parse_keywords(history_entry.get("keywords"))
        heatmap_data_uri = comparison_service.build_history_heatmap_data_uri(keywords, matches)

        return render_template(
            "history_detail.html",
            user=user,
            history=history_entry,
            matches=matches,
            keywords=keywords,
            heatmap_data_uri=heatmap_data_uri,
        )

    @app.route("/admin/researchers")
    @role_required("Admin")
    def manage_researchers():
        return render_template(
            "manage_researchers.html",
            user=get_current_user(),
            researchers=admin_service.list_researchers(),
        )

    @app.route("/admin/researchers/<int:researcher_id>/delete", methods=["POST"])
    @role_required("Admin")
    def admin_delete_researcher(researcher_id):
        deleted = admin_service.delete_researcher(researcher_id)
        if deleted:
            flash(f"Researcher ID {researcher_id} has been deleted.", "info")
        else:
            flash("Researcher not found.", "warning")
        return redirect(url_for("manage_researchers"))

    @app.route("/admin/researchers/<int:researcher_id>/history")
    @role_required("Admin")
    def admin_view_history(researcher_id):
        researcher, history_rows = admin_service.get_researcher_history(researcher_id)
        if not researcher:
            flash("Researcher not found.", "warning")
            return redirect(url_for("manage_researchers"))

        return render_template(
            "history.html",
            user=get_current_user(),
            history_rows=history_rows,
            selected_researcher=researcher,
        )

    @app.route("/admin/researchers/<int:researcher_id>/reset", methods=["GET", "POST"])
    @role_required("Admin")
    def admin_reset_password(researcher_id):
        researcher = users_repo.get_researcher_by_id(researcher_id)
        if not researcher:
            flash("Researcher not found.", "danger")
            return redirect(url_for("manage_researchers"))

        if request.method == "GET":
            return render_template(
                "admin_reset_password.html",
                user=get_current_user(),
                researcher=researcher,
            )

        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")
        error = admin_service.validate_password_reset(new_password, confirm_password)
        if error:
            flash(error[0], error[1])
            return redirect(url_for("admin_reset_password", researcher_id=researcher_id))

        admin_service.reset_researcher_password(researcher_id, new_password)
        flash(
            f"Password updated for {researcher['first_name']} {researcher['last_name']}.",
            "success",
        )
        return redirect(url_for("manage_researchers"))

    @app.route("/history/<int:history_id>/heatmap")
    @login_required
    def history_heatmap(history_id):
        user = get_current_user()
        history_entry, matches = matcher.get_history_with_matches(history_id)
        if not history_entry:
            flash("History entry not found.", "danger")
            return redirect(url_for("history"))

        if user["role"] == "Researcher" and history_entry["researcher_id"] != user["id"]:
            flash("You are not allowed to view that history entry.", "danger")
            return redirect(url_for("history"))

        keywords = parse_keywords(history_entry.get("keywords"))
        if not keywords or not matches:
            flash("Not enough data to build a heatmap for this entry.", "warning")
            return redirect(url_for("history_detail", history_id=history_id))

        table_data = comparison_service.build_history_heatmap_table(keywords, matches)
        if not table_data:
            flash("Unable to build heatmap matrix for this entry.", "warning")
            return redirect(url_for("history_detail", history_id=history_id))

        return render_template(
            "history_heatmap_table.html",
            user=user,
            history=history_entry,
            col_labels=table_data["col_labels"],
            table_rows=table_data["table_rows"],
            min_val=table_data["min_val"],
            max_val=table_data["max_val"],
        )
