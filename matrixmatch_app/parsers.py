import json


def parse_keywords(raw_keywords):
    """Parse keywords from JSON array string, comma-separated string, or list."""
    if raw_keywords is None:
        return []

    if isinstance(raw_keywords, list):
        return [str(k).strip() for k in raw_keywords if str(k).strip()]

    if not isinstance(raw_keywords, str):
        raw_keywords = str(raw_keywords)

    raw_keywords = raw_keywords.strip()
    if not raw_keywords:
        return []

    if raw_keywords.startswith("["):
        try:
            parsed = json.loads(raw_keywords)
            if isinstance(parsed, list):
                return [str(k).strip() for k in parsed if str(k).strip()]
            return [part.strip() for part in str(parsed).split(",") if part.strip()]
        except json.JSONDecodeError:
            pass

    return [part.strip() for part in raw_keywords.split(",") if part.strip()]
