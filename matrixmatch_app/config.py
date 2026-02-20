import os
from pathlib import Path

_ENV_LOADED = False


def _load_env_file():
    """Load project-level .env values once if present."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

    _ENV_LOADED = True


def get_db_config():
    """Build DB config from environment variables."""
    _load_env_file()
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "dbname": os.getenv("DB_NAME", "matrixmatch"),
        "options": os.getenv("DB_OPTIONS", "-c search_path=matrixmatch,public"),
    }


def get_secret_key():
    _load_env_file()
    return os.getenv("SECRET_KEY", "supersecretkey_change_me")
