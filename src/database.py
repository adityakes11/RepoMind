import os
import uuid
import psycopg
from dotenv import load_dotenv
load_dotenv()

def get_connection():
    return psycopg.connect(host=os.getenv("POSTGRES_HOST","localhost"),port=os.getenv("POSTGRES_PORT","5432"),dbname=os.getenv("POSTGRES_DB","repomind"),user=os.getenv("POSTGRES_USER","postgres"),password=os.getenv("POSTGRES_PASSWORD",""))

def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS threads (id UUID PRIMARY KEY,title TEXT NOT NULL,github_url TEXT,repo_name TEXT,collection_name TEXT,created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP);""")
            cur.execute("""CREATE TABLE IF NOT EXISTS messages (id BIGSERIAL PRIMARY KEY,thread_id UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,role TEXT NOT NULL,content TEXT NOT NULL,created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP);""")

def create_thread(title="New Repository"):
    tid=uuid.uuid4()
    with get_connection() as conn:
        with conn.cursor() as cur: cur.execute("INSERT INTO threads (id,title) VALUES (%s,%s)",(tid,title))
    return tid

def get_threads():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id,title,repo_name,updated_at
                FROM threads
                WHERE collection_name IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM messages
                      WHERE messages.thread_id = threads.id AND role = 'user'
                  )
                  AND EXISTS (
                      SELECT 1 FROM messages
                      WHERE messages.thread_id = threads.id AND role = 'assistant'
                  )
                ORDER BY updated_at DESC
            """)
            rows=cur.fetchall()
    return [{"id":r[0],"title":r[1],"repo_name":r[2],"updated_at":r[3]} for r in rows]

def get_thread(tid):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id,title,github_url,repo_name,collection_name FROM threads WHERE id=%s",(tid,)); r=cur.fetchone()
            if not r:return None
            cur.execute("SELECT role,content,created_at FROM messages WHERE thread_id=%s ORDER BY created_at ASC,id ASC",(tid,)); ms=cur.fetchall()
    return {"id":r[0],"title":r[1],"github_url":r[2],"repo_name":r[3],"collection_name":r[4],"messages":[{"role":m[0],"content":m[1],"created_at":m[2]} for m in ms]}

def update_thread_repository(tid,url,repo_name,collection_name):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE threads SET title=%s,github_url=%s,repo_name=%s,collection_name=%s,updated_at=CURRENT_TIMESTAMP WHERE id=%s",(repo_name or "Repository",url,repo_name,collection_name,tid))

def save_message(tid,role,content):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO messages (thread_id,role,content) VALUES (%s,%s,%s)",(tid,role,content))
            cur.execute("UPDATE threads SET updated_at=CURRENT_TIMESTAMP WHERE id=%s",(tid,))
