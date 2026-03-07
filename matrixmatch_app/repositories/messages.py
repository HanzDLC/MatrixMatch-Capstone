import logging
from matrixmatch_app.db import db_cursor

logger = logging.getLogger(__name__)

def send_message(sender_id: int, receiver_id: int, content: str) -> None:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            INSERT INTO matrixmatch.messages (sender_id, receiver_id, content)
            VALUES (%s, %s, %s)
            """,
            (sender_id, receiver_id, content),
        )

def get_conversation(user1_id: int, user2_id: int) -> list:
    """Gets the chronological message history between two specific users."""
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT m.id, m.sender_id, m.receiver_id, m.content, m.timestamp, m.is_read
            FROM matrixmatch.messages m
            WHERE (m.sender_id = %s AND m.receiver_id = %s)
               OR (m.sender_id = %s AND m.receiver_id = %s)
            ORDER BY m.timestamp ASC
            """,
            (user1_id, user2_id, user2_id, user1_id),
        )
        return cursor.fetchall()

def get_recent_conversations(user_id: int) -> list:
    """
    Gets a summary of all recent conversations for a user, 
    returning the latest message and the other user's details.
    """
    with db_cursor() as cursor:
        cursor.execute(
            """
            WITH ranked_messages AS (
                SELECT 
                    m.id, m.sender_id, m.receiver_id, m.content, m.timestamp, m.is_read,
                    CASE WHEN m.sender_id = %s THEN m.receiver_id ELSE m.sender_id END as other_user_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY CASE WHEN m.sender_id = %s THEN m.receiver_id ELSE m.sender_id END
                        ORDER BY m.timestamp DESC
                    ) as rn
                FROM matrixmatch.messages m
                WHERE m.sender_id = %s OR m.receiver_id = %s
            )
            SELECT 
                rm.id, rm.sender_id, rm.receiver_id, rm.content, rm.timestamp, rm.is_read,
                u.researcher_id as other_user_id, u.first_name, u.last_name, u.profile_pic
            FROM ranked_messages rm
            JOIN matrixmatch."user" u ON rm.other_user_id = u.researcher_id
            WHERE rm.rn = 1
            ORDER BY rm.timestamp DESC
            """,
            (user_id, user_id, user_id, user_id),
        )
        return cursor.fetchall()
        
def get_unread_count(user_id: int) -> int:
    """Returns the total number of unread messages sent TO this user."""
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM matrixmatch.messages
            WHERE receiver_id = %s AND is_read = FALSE
            """,
            (user_id,)
        )
        row = cursor.fetchone()
        return row["count"] if row else 0

def mark_as_read(sender_id: int, receiver_id: int) -> None:
    """Marks all messages from sender_id to receiver_id as read."""
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE matrixmatch.messages
            SET is_read = TRUE
            WHERE sender_id = %s AND receiver_id = %s AND is_read = FALSE
            """,
            (sender_id, receiver_id)
        )
