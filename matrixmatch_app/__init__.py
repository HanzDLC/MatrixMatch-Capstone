"""Application factory for MatrixMatch."""

from pathlib import Path

from flask import Flask

from matrixmatch_app.config import get_secret_key
from matrixmatch_app.routes import register_routes


def create_app() -> Flask:
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.config["SECRET_KEY"] = get_secret_key()

    register_routes(app)
    return app
