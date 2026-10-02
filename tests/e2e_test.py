"""Live end-to-end validation for the authenticated RepoMind API.

Run with:
    python tests/e2e_test.py

Environment overrides:
    E2E_BASE_URL=http://127.0.0.1:8000
    E2E_REPO_URL=https://github.com/adityakes11/YoutubeRAGChatbot
    E2E_EMAIL=repomind.e2e@example.com
    E2E_PASSWORD=SecureTest123!
    E2E_INGEST_TIMEOUT=900
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


BASE_URL = os.getenv("E2E_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REPO_URL = os.getenv("E2E_REPO_URL", "https://github.com/adityakes11/YoutubeRAGChatbot")
EMAIL = os.getenv("E2E_EMAIL", "repomind.e2e@example.com")
PASSWORD = os.getenv("E2E_PASSWORD", "SecureTest123!")
INGEST_TIMEOUT = int(os.getenv("E2E_INGEST_TIMEOUT", "900"))


class E2EFailure(RuntimeError):
    pass


def request_json(method, path, payload=None, token=None, expected=None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{BASE_URL}{path}", data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    try:
        result = json.loads(raw.decode("utf-8")) if raw else None
    except json.JSONDecodeError:
        result = raw.decode("utf-8", errors="replace")
    if expected is not None and status not in expected:
        raise E2EFailure(f"{method} {path}: expected {expected}, got {status}: {result}")
    return status, result


def stream_answer(thread_id, token, question):
    query = urllib.parse.urlencode({"question": question})
    request = urllib.request.Request(
        f"{BASE_URL}/api/threads/{thread_id}/chat/stream?{query}",
        headers={"Accept": "text/event-stream", "Authorization": f"Bearer {token}"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=INGEST_TIMEOUT)
    except urllib.error.HTTPError as exc:
        raise E2EFailure(f"GET chat stream: HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc

    chunks = []
    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:].strip())
            chunks.append(event.get("content", ""))
    answer = "".join(chunks).strip()
    if not answer:
        raise E2EFailure("GET chat stream returned no answer content")
    return answer


def run():
    print(f"[1/8] Checking health at {BASE_URL}")
    _, health = request_json("GET", "/api/health", expected={200})
    if health.get("database") != "ok" or health.get("ollama") != "ok":
        raise E2EFailure(f"Dependencies are not ready: {health}")

    print(f"[2/8] Signing up or logging in as {EMAIL}")
    status, session = request_json(
        "POST", "/api/auth/signup", {"email": EMAIL, "password": PASSWORD}, expected={201, 409}
    )
    if status == 409:
        _, session = request_json(
            "POST", "/api/auth/login", {"email": EMAIL, "password": PASSWORD}, expected={200}
        )
    token = session["access_token"]

    print("[3/8] Verifying the authenticated identity")
    _, user = request_json("GET", "/api/auth/me", token=token, expected={200})
    if user["email"] != EMAIL.lower():
        raise E2EFailure(f"Authenticated identity mismatch: {user}")

    thread = None
    try:
        print("[4/8] Creating an isolated workspace thread")
        _, thread = request_json(
            "POST", "/api/threads", {"title": "YoutubeRAGChatbot E2E"}, token, expected={201}
        )

        print(f"[5/8] Ingesting {REPO_URL}")
        _, ingest = request_json(
            "POST", "/api/repos/ingest", {"github_url": REPO_URL, "thread_id": thread["id"]}, token, expected={202}
        )
        deadline = time.monotonic() + INGEST_TIMEOUT
        while time.monotonic() < deadline:
            _, repository = request_json(
                "GET", f"/api/repos/{ingest['collection_name']}/status", token=token, expected={200}
            )
            current_status = repository["status"]
            print(f"    status={current_status} progress={repository.get('progress', 0)}%", end="\r")
            if current_status in {"completed", "ready"}:
                break
            if current_status == "failed":
                raise E2EFailure(f"Repository ingestion failed: {repository.get('error')}")
            time.sleep(2)
        else:
            raise E2EFailure(f"Repository ingestion timed out after {INGEST_TIMEOUT}s")
        print()

        print("[6/8] Asking the indexed repository through authenticated SSE")
        answer = stream_answer(
            thread["id"], token, "What is the main purpose of this repository and which file starts the application?"
        )
        print(f"    answer preview: {answer[:160].replace(chr(10), ' ')}")

        print("[7/8] Verifying persisted chat history")
        _, saved_thread = request_json("GET", f"/api/threads/{thread['id']}", token=token, expected={200})
        roles = [message["role"] for message in saved_thread["messages"]]
        if roles[-2:] != ["user", "assistant"]:
            raise E2EFailure(f"Unexpected persisted roles: {roles}")
    finally:
        if thread:
            request_json("DELETE", f"/api/threads/{thread['id']}", token=token, expected={204})

    print("[8/8] Revoking the session")
    request_json("POST", "/api/auth/logout", token=token, expected={204})
    status, _ = request_json("GET", "/api/threads", token=token, expected={401})
    if status != 401:
        raise E2EFailure("Revoked session remained usable")
    print("E2E PASS: authentication, isolation, ingestion, SSE chat, persistence, and logout")


if __name__ == "__main__":
    try:
        run()
    except (E2EFailure, urllib.error.URLError) as exc:
        raise SystemExit(f"E2E FAIL: {exc}") from exc