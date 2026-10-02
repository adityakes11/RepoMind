import os
import uuid
import psycopg
from dotenv import load_dotenv
from src.config import settings
load_dotenv()

def get_connection():
    return psycopg.connect(
        host=settings.postgres.host,
        port=settings.postgres.port,
        dbname=settings.postgres.db,
        user=settings.postgres.user,
        password=settings.postgres.password,
        connect_timeout=int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5")),
    )

def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS users (id UUID PRIMARY KEY,email TEXT NOT NULL UNIQUE,password_hash TEXT NOT NULL,created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP);""")
            cur.execute("""CREATE TABLE IF NOT EXISTS sessions (id UUID PRIMARY KEY,user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,token_hash TEXT NOT NULL UNIQUE,expires_at TIMESTAMPTZ NOT NULL,created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP);""")
            cur.execute("""CREATE TABLE IF NOT EXISTS threads (id UUID PRIMARY KEY,title TEXT NOT NULL,github_url TEXT,repo_name TEXT,collection_name TEXT,created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP);""")
            cur.execute("ALTER TABLE threads ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE CASCADE")
            cur.execute("CREATE INDEX IF NOT EXISTS sessions_token_hash_idx ON sessions(token_hash)")
            cur.execute("CREATE INDEX IF NOT EXISTS threads_user_id_idx ON threads(user_id)")
            cur.execute("""CREATE TABLE IF NOT EXISTS messages (id BIGSERIAL PRIMARY KEY,thread_id UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,role TEXT NOT NULL,content TEXT NOT NULL,created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP);""")

def create_user(email,password_hash):
    user_id=uuid.uuid4()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (id,email,password_hash) VALUES (%s,%s,%s)",(user_id,email,password_hash))
    return user_id

def get_user_by_email(email):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id,email,password_hash,created_at FROM users WHERE email=%s",(email,)); row=cur.fetchone()
    return {"id":row[0],"email":row[1],"password_hash":row[2],"created_at":row[3]} if row else None

def get_user_by_id(user_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id,email,created_at FROM users WHERE id=%s",(user_id,)); row=cur.fetchone()
    return {"id":row[0],"email":row[1],"created_at":row[2]} if row else None

def create_session(user_id,token_hash,expires_at):
    session_id=uuid.uuid4()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO sessions (id,user_id,token_hash,expires_at) VALUES (%s,%s,%s,%s)",(session_id,user_id,token_hash,expires_at))
    return session_id

def get_user_by_token_hash(token_hash):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT u.id,u.email,u.created_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=%s AND s.expires_at>CURRENT_TIMESTAMP""",(token_hash,)); row=cur.fetchone()
    return {"id":row[0],"email":row[1],"created_at":row[2]} if row else None

def delete_session(token_hash):
    with get_connection() as conn:
        with conn.cursor() as cur: cur.execute("DELETE FROM sessions WHERE token_hash=%s",(token_hash,))

def create_thread(user_id,title="New Repository"):
    tid=uuid.uuid4()
    with get_connection() as conn:
        with conn.cursor() as cur: cur.execute("INSERT INTO threads (id,user_id,title) VALUES (%s,%s,%s)",(tid,user_id,title))
    return tid

def get_threads(user_id):
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
                AND user_id=%s
                ORDER BY updated_at DESC
            """,(user_id,))
            rows=cur.fetchall()
    return [{"id":r[0],"title":r[1],"repo_name":r[2],"updated_at":r[3]} for r in rows]

def get_all_threads(user_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT t.id,t.title,t.repo_name,t.github_url,t.collection_name,t.updated_at,
                       (SELECT content FROM messages m WHERE m.thread_id=t.id ORDER BY m.created_at DESC,m.id DESC LIMIT 1)
                FROM threads t
                WHERE t.user_id=%s
                ORDER BY t.updated_at DESC
            """,(user_id,))
            rows=cur.fetchall()
    return [{"id":r[0],"title":r[1],"repo_name":r[2],"github_url":r[3],"collection_name":r[4],"updated_at":r[5],"last_message":r[6]} for r in rows]

def get_thread(tid,user_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id,title,github_url,repo_name,collection_name,updated_at FROM threads WHERE id=%s AND user_id=%s",(tid,user_id)); r=cur.fetchone()
            if not r:return None
            cur.execute("SELECT role,content,created_at FROM messages WHERE thread_id=%s ORDER BY created_at ASC,id ASC",(tid,)); ms=cur.fetchall()
    return {"id":r[0],"title":r[1],"github_url":r[2],"repo_name":r[3],"collection_name":r[4],"updated_at":r[5],"messages":[{"role":m[0],"content":m[1],"created_at":m[2]} for m in ms]}

def update_thread_repository(tid,user_id,url,repo_name,collection_name):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE threads SET title=%s,github_url=%s,repo_name=%s,collection_name=%s,updated_at=CURRENT_TIMESTAMP WHERE id=%s AND user_id=%s",(repo_name or "Repository",url,repo_name,collection_name,tid,user_id))

def save_message(tid,user_id,role,content):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO messages (thread_id,role,content) SELECT %s,%s,%s WHERE EXISTS (SELECT 1 FROM threads WHERE id=%s AND user_id=%s)",(tid,role,content,tid,user_id))
            if cur.rowcount == 0: raise ValueError("Thread not found")
            cur.execute("UPDATE threads SET updated_at=CURRENT_TIMESTAMP WHERE id=%s AND user_id=%s",(tid,user_id))

def delete_thread(tid,user_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM threads WHERE id=%s AND user_id=%s",(tid,user_id))
            return cur.rowcount > 0
