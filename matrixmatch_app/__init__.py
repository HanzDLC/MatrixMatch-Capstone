"""Application factory for MatrixMatch."""

from flask import Flask

from matrixmatch_app.config import get_secret_key
from matrixmatch_app.routes import register_routes


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = get_secret_key()

    register_routes(app)
    return app
