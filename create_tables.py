from matrixmatch_app.db import db_cursor

def create_documents_log_table():
    with db_cursor(commit=True) as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matrixmatch.documents_log (
                log_id SERIAL PRIMARY KEY,
                document_id INT,
                document_title VARCHAR(255) NOT NULL,
                action VARCHAR(50) NOT NULL,
                modified_by VARCHAR(255) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("documents_log table created successfully!")

if __name__ == "__main__":
    create_documents_log_table()
