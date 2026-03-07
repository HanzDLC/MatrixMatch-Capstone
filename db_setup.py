from matrixmatch_app.db import db_cursor

try:
    with db_cursor(commit=True) as cursor:
        cursor.execute('ALTER TABLE matrixmatch."user" ADD COLUMN IF NOT EXISTS last_seen TIMESTAMP WITH TIME ZONE;')
        print("last_seen column added.")
except Exception as e:
    print(f"Error: {e}")
