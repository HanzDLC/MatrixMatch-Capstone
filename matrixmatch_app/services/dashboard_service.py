from typing import Dict, Optional

from matrixmatch_app.repositories import history, users


def get_admin_dashboard_data(admin_user_id: int) -> Optional[Dict]:
    admin_user = users.get_user_by_id(admin_user_id)
    if not admin_user or admin_user.get("role") != "Admin":
        return None

    return {
        "user": admin_user,
        "stats": history.get_admin_stats(),
        "recent_history": history.list_recent_history(limit=10),
    }


def get_researcher_dashboard_data(researcher_id: int) -> Dict:
    stats = history.get_researcher_stats(researcher_id) or {}
    avg_threshold = float(stats.get("avg_threshold") or 0) * 100
    return {
        "stats": {
            "total_comparisons": int(stats.get("total_comparisons") or 0),
            "avg_threshold_pct": round(avg_threshold, 1),
            "last_7_days_runs": int(stats.get("last_7_days_runs") or 0),
        },
        "recent_history": history.list_recent_history_for_user(researcher_id, limit=5),
    }
