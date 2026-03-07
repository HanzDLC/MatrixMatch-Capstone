import importlib
import logging

from flask import flash, redirect, render_template, request, session, url_for

from matrixmatch_app.auth import get_current_user, login_required, role_required
from matrixmatch_app.parsers import parse_keywords
from matrixmatch_app.repositories import history as history_repo, document_logs as document_logs_repo, users as users_repo
from matrixmatch_app.services import admin_service, auth_service, comparison_service, dashboard_service, document_service

logger = logging.getLogger(__name__)


def _get_matcher_module():
    try:
        return importlib.import_module("matcher")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Comparison dependencies are not installed. Install requirements.txt to enable comparison and history features."
        ) from exc


def _build_history_detail_extras(matcher_module, history_entry, matches):
    keywords = parse_keywords(history_entry.get("keywords"))

    semantic_highlights = []
    if matches:
        try:
            semantic_highlights = matcher_module.build_semantic_sentence_highlights(
                user_abstract=history_entry.get("user_abstract", ""),
                matches=matches,
            )
        except Exception:
            logger.exception(
                "Failed to build semantic highlights for history %s.",
                history_entry.get("history_id"),
            )

    table_data = None
    if keywords and matches:
        table_data = comparison_service.build_history_heatmap_table(keywords, matches)

    return keywords, semantic_highlights, table_data


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
        return redirect(url_for("home"))

    @app.route("/guest")
    def guest_login():
        session.clear()
        session["is_guest"] = True
        flash("You are browsing as a Guest. Comparisons will not be saved.", "info")
        return redirect(url_for("dashboard"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        if session.get("is_guest"):
            user = get_current_user()
            guest_history_dict = session.get("guest_history", {})
            # Convert dict values to a list and sort by history_id desc (which is timestamp-based)
            recent_history = sorted(
                guest_history_dict.values(),
                key=lambda x: x.get("history_id", 0),
                reverse=True
            )
            
            return render_template(
                "dashboard_researcher.html",
                user=user,
                stats={
                    "total_comparisons": len(recent_history),
                    "avg_threshold_pct": round(sum(h["similarity_threshold"] for h in recent_history) * 100 / len(recent_history), 1) if recent_history else 0,
                    "last_7_days_runs": len(recent_history) # All are recent in a session
                },
                recent_history=recent_history[:5],
            )

        role = session.get("role", "")
        if role == "Admin":
            return redirect(url_for("admin_dashboard"))
        if role == "Researcher":
            return redirect(url_for("researcher_dashboard"))

        session.clear()
        return redirect(url_for("login"))

    @app.route("/profile", methods=["GET", "POST"])
    @login_required
    def profile():
        if session.get("is_guest"):
            flash("Guests cannot edit profiles. Please create an account.", "warning")
            return redirect(url_for("dashboard"))

        user = get_current_user()

        if request.method == "GET":
            return render_template("profile.html", user=user)

        first_name = request.form.get("first_name", "")
        last_name = request.form.get("last_name", "")
        old_password = request.form.get("old_password", "")
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        success, message = auth_service.update_profile(
            user["id"], first_name, last_name, old_password, password, confirm_password
        )

        if success:
            session["first_name"] = first_name
            session["last_name"] = last_name
            flash(message[0], message[1])
            return redirect(url_for("profile"))
        
        flash(message[0], message[1])
        return render_template(
            "profile.html",
            user={**user, "first_name": first_name, "last_name": last_name}
        )

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
            stats=data.get("stats", {}),
            recent_history=data["recent_history"],
        )

    @app.route("/comparison/new", methods=["GET", "POST"])
    @login_required
    def comparison_new():
        user = get_current_user()
        if request.method == "GET":
            return render_template("comparison_new.html", user=user)

        # --- Guest path: skip DB save ---
        if session.get("is_guest"):
            raw_keywords = (request.form.get("keywords", "") or "").strip()
            user_abstract = (request.form.get("abstract", "") or "").strip()
            program_filter = (request.form.get("program_filter", "ALL") or "ALL").strip() or "ALL"
            threshold_str = request.form.get("threshold", "60")

            if not raw_keywords or not user_abstract:
                flash("Please enter both keywords and an abstract.", "danger")
                return redirect(url_for("comparison_new"))

            keywords = parse_keywords(raw_keywords)
            if len(keywords) < 5:
                flash("Please enter at least 5 keywords.", "danger")
                return redirect(url_for("comparison_new"))

            similarity_threshold = comparison_service.parse_threshold(threshold_str)
            try:
                matcher = _get_matcher_module()
            except RuntimeError as exc:
                flash(str(exc), "danger")
                return redirect(url_for("comparison_new"))

            try:
                history_id, _matches, history_data = matcher.run_stage1_guest(
                    keywords=keywords,
                    user_abstract=user_abstract,
                    academic_program_filter=program_filter,
                    similarity_threshold=similarity_threshold,
                )
            except Exception:
                logger.exception("Failed to run guest comparison.")
                flash("Unable to run the comparison right now. Please try again shortly.", "danger")
                return redirect(url_for("comparison_new"))
            if history_id is None:
                flash("No documents found for the selected program.", "warning")
                return redirect(url_for("comparison_new"))

            # Store in session
            guest_history = session.get("guest_history", {})
            guest_history[str(history_id)] = history_data
            session["guest_history"] = guest_history

            flash("Stage 1 comparison completed (Guest mode — not saved).", "success")
            return redirect(url_for("history_detail", history_id=history_id))

        # --- Normal (logged-in) path ---
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

        if session.get("is_guest"):
            guest_history_dict = session.get("guest_history", {})
            history_rows = sorted(
                guest_history_dict.values(),
                key=lambda x: x.get("history_id", 0),
                reverse=True
            )
            return render_template(
                "history.html",
                user=user,
                history_rows=history_rows,
            )

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

        # --- Guest path: read from session ---
        if session.get("is_guest"):
            guest_history = session.get("guest_history", {})
            history_data = guest_history.get(str(history_id))
            if not history_data:
                flash("Guest history entry not found.", "warning")
                return redirect(url_for("comparison_new"))

            # Rebuild matches from top_matches string
            history_entry = dict(history_data)
            history_entry["keywords_list"] = parse_keywords(history_entry.get("keywords"))
            keywords = history_entry["keywords_list"]
            try:
                matcher = _get_matcher_module()
            except RuntimeError as exc:
                flash(str(exc), "danger")
                return redirect(url_for("comparison_new"))

            doc_pairs = matcher._parse_top_matches(history_entry.get("top_matches", ""))
            matches = []
            if doc_pairs:
                from matrixmatch_app.db import db_cursor
                doc_ids = [p[0] for p in doc_pairs]
                placeholders = ", ".join(["%s"] * len(doc_ids))
                with db_cursor() as cursor:
                    cursor.execute(
                        f"SELECT document_id, title, academic_program, abstract FROM matrixmatch.documents WHERE document_id IN ({placeholders})",
                        tuple(doc_ids),
                    )
                    docs = cursor.fetchall()
                docs_by_id = {row["document_id"]: row for row in docs}
                for doc_id, similarity in doc_pairs:
                    doc = docs_by_id.get(doc_id)
                    if doc:
                        matches.append({
                            "document_id": doc["document_id"],
                            "title": doc["title"],
                            "program": doc.get("academic_program") or "",
                            "similarity": similarity,
                            "abstract": doc.get("abstract") or "",
                        })

            keywords, semantic_highlights, table_data = _build_history_detail_extras(
                matcher,
                history_entry,
                matches,
            )

            return render_template(
                "history_detail.html",
                user=user,
                history=history_entry,
                matches=matches,
                keywords=keywords,
                semantic_highlights=semantic_highlights,
                table_data=table_data,
            )

        try:
            matcher = _get_matcher_module()
        except RuntimeError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("history"))

        # --- Normal path ---
        try:
            history_entry, matches = matcher.get_history_with_matches(history_id)
        except Exception:
            logger.exception("Failed to load history detail %s.", history_id)
            flash("Unable to load this history entry right now.", "danger")
            return redirect(url_for("history"))

        if not history_entry:
            flash("History entry not found.", "warning")
            return redirect(url_for("history"))

        if user["role"] == "Researcher" and history_entry["researcher_id"] != user["id"]:
            flash("You are not allowed to view that history entry.", "danger")
            return redirect(url_for("history"))

        keywords, semantic_highlights, table_data = _build_history_detail_extras(
            matcher,
            history_entry,
            matches,
        )

        return render_template(
            "history_detail.html",
            user=user,
            history=history_entry,
            matches=matches,
            keywords=keywords,
            semantic_highlights=semantic_highlights,
            table_data=table_data,
        )

    @app.route("/admin/researchers")
    @role_required("Admin")
    def manage_researchers():
        dashboard_data = dashboard_service.get_admin_dashboard_data(session["user_id"]) or {}
        return render_template(
            "manage_researchers.html",
            user=get_current_user(),
            researchers=admin_service.list_researchers(),
            protected_admin_email=admin_service.PROTECTED_ADMIN_EMAIL,
            stats=dashboard_data.get("stats", {}),
            recent_history=dashboard_data.get("recent_history", []),
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

    @app.route("/admin/researchers/<int:researcher_id>/promote", methods=["POST"])
    @role_required("Admin")
    def admin_promote_researcher(researcher_id):
        promoted = admin_service.promote_researcher(researcher_id)
        if promoted:
            flash(f"Researcher ID {researcher_id} has been promoted to Admin.", "success")
        else:
            flash("Researcher not found or could not be promoted.", "warning")
        return redirect(url_for("manage_researchers"))

    @app.route("/admin/researchers/<int:researcher_id>/demote", methods=["POST"])
    @role_required("Admin")
    def admin_demote_researcher(researcher_id):
        demoted, outcome = admin_service.demote_admin(researcher_id)
        flash(outcome[0], outcome[1])

        if demoted and session.get("user_id") == researcher_id:
            session["role"] = "Researcher"
            return redirect(url_for("dashboard"))

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
        researcher = users_repo.get_user_by_id(researcher_id)
        if not researcher:
            flash("User not found.", "danger")
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
        try:
            matcher = _get_matcher_module()
        except RuntimeError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("history"))

        try:
            history_entry, matches = matcher.get_history_with_matches(history_id)
        except Exception:
            logger.exception("Failed to load history heatmap %s.", history_id)
            flash("Unable to load the stage 2 heatmap right now.", "danger")
            return redirect(url_for("history"))

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

    @app.route("/admin/documents")
    @role_required("Admin")
    def manage_documents():
        documents = document_service.list_all_documents()
        return render_template(
            "manage_documents.html",
            user=get_current_user(),
            documents=documents,
        )

    @app.route("/admin/documents/new", methods=["GET", "POST"])
    @role_required("Admin")
    def add_document():
        if request.method == "GET":
            return render_template("document_form.html", user=get_current_user(), document=None)

        title = request.form.get("title", "")
        program = request.form.get("academic_program", "")
        abstract = request.form.get("abstract", "")

        user = get_current_user()
        modified_by = f"{user['first_name']} {user['last_name']}"

        success, message = document_service.add_document(title, program, abstract, modified_by)
        if success:
            category = "warning" if "audit log" in message.lower() else "success"
            flash(message, category)
            return redirect(url_for("manage_documents"))
        else:
            flash(message, "danger")
            return render_template(
                "document_form.html",
                user=get_current_user(),
                document={"title": title, "academic_program": program, "abstract": abstract}
            )

    @app.route("/admin/documents/<int:document_id>/edit", methods=["GET", "POST"])
    @role_required("Admin")
    def edit_document(document_id):
        document = document_service.get_document(document_id)
        if not document:
            flash("Document not found.", "warning")
            return redirect(url_for("manage_documents"))

        if request.method == "GET":
            return render_template("document_form.html", user=get_current_user(), document=document)

        title = request.form.get("title", "")
        program = request.form.get("academic_program", "")
        abstract = request.form.get("abstract", "")

        user = get_current_user()
        modified_by = f"{user['first_name']} {user['last_name']}"

        success, message = document_service.update_document(document_id, title, program, abstract, modified_by)
        if success:
            category = "warning" if "audit log" in message.lower() else "success"
            flash(message, category)
            return redirect(url_for("manage_documents"))
        else:
            flash(message, "danger")
            return render_template(
                "document_form.html",
                user=get_current_user(),
                document={"document_id": document_id, "title": title, "academic_program": program, "abstract": abstract}
            )

    @app.route("/admin/documents/<int:document_id>/delete", methods=["POST"])
    @role_required("Admin")
    def admin_delete_document(document_id):
        user = get_current_user()
        modified_by = f"{user['first_name']} {user['last_name']}"
        deleted = document_service.delete_document(document_id, modified_by)
        if deleted:
            flash(f"Document ID {document_id} has been deleted.", "info")
        else:
            flash("Document not found.", "warning")
        return redirect(url_for("manage_documents"))

    @app.route("/admin/document-logs")
    @role_required("Admin")
    def document_logs():
        logs = document_logs_repo.get_all_logs()
        return render_template(
            "document_logs.html",
            user=get_current_user(),
            logs=logs,
        )

    @app.route("/comparison/extract", methods=["POST"])
    @login_required
    def comparison_extract_document():
        if "file" not in request.files:
            return {"error": "No file uploaded"}, 400
        file = request.files["file"]
        if not file or not file.filename:
            return {"error": "No selected file"}, 400
        file_data = file.read()
        success, message, data = document_service.extract_document_info(file_data, file.filename)
        if success:
            import re
            from collections import Counter
            
            combined_text = (data.get("title", "") + " " + data.get("abstract", "")).lower()
            words = re.findall(r'\b[a-z]{4,}\b', combined_text)
            stopwords = {"this", "that", "with", "from", "which", "were", "study", "research", "system", "using", "based", "results", "data", "analysis"}
            filtered_words = [w for w in words if w not in stopwords]
            top_words = [word for word, count in Counter(filtered_words).most_common(15)]
            
            return {
                "title": data.get("title", ""),
                "abstract": data.get("abstract", ""),
                "keywords": top_words
            }, 200
        else:
            return {"error": message}, 400

    @app.route("/admin/documents/extract", methods=["POST"])
    @role_required("Admin")
    def admin_extract_document():
        if "file" not in request.files:
            return {"error": "No file uploaded"}, 400
            
        file = request.files["file"]
        if not file or not file.filename:
            return {"error": "No selected file"}, 400
            
        file_data = file.read()
        success, message, data = document_service.extract_document_info(file_data, file.filename)
        
        if success:
            return {"title": data.get("title", ""), "abstract": data.get("abstract", "")}, 200
        else:
            return {"error": message}, 400
