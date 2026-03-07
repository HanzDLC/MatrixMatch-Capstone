import sys
from matrixmatch_app.db import db_cursor
try:
    with db_cursor() as cursor:
        cursor.execute("SELECT researcher_id, first_name, last_seen, NOW() as current_time FROM matrixmatch.\"user\"")
        for row in cursor.fetchall():
            print(f"User: {row['first_name']}, Last Seen: {row['last_seen']}, NOW: {row['current_time']}")
except Exception as e:
    print('Error:', e)
