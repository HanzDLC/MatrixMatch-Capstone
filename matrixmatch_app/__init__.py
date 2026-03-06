"""Application factory for MatrixMatch."""

from pathlib import Path

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from matrixmatch_app.config import get_secret_key
from matrixmatch_app.routes import register_routes


def create_app() -> Flask:
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    # Respect forwarded host/proto when the app sits behind Vercel/ngrok.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    app.config["SECRET_KEY"] = get_secret_key()

    register_routes(app)
    return app
