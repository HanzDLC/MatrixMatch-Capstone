import os


def get_db_config():
    """Build DB config from environment variables."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "dbname": os.getenv("DB_NAME", "matrixmatch"),
        "options": os.getenv("DB_OPTIONS", "-c search_path=matrixmatch,public"),
    }


def get_secret_key():
    return os.getenv("SECRET_KEY", "supersecretkey_change_me")
