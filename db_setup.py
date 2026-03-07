from matrixmatch_app.db import db_cursor

try:
    with db_cursor(commit=True) as cursor:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matrixmatch.messages (
                id SERIAL PRIMARY KEY,
                sender_id INTEGER NOT NULL REFERENCES matrixmatch."user"(researcher_id) ON DELETE CASCADE,
                receiver_id INTEGER NOT NULL REFERENCES matrixmatch."user"(researcher_id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                is_read BOOLEAN DEFAULT FALSE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_sender ON matrixmatch.messages(sender_id);
            CREATE INDEX IF NOT EXISTS idx_messages_receiver ON matrixmatch.messages(receiver_id);
        ''')
        print("messages table added.")
except Exception as e:
    print(f"Error: {e}")
