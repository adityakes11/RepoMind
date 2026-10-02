import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from src.database import create_session, create_user, get_user_by_email, get_user_by_token_hash


SESSION_DAYS = 30


def normalize_email(email):
    normalized = email.strip().lower()
    if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
        raise ValueError("Enter a valid email address")
    return normalized


def hash_password(password):
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$16384$8$1${}${}".format(
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(derived).decode("ascii"),
    )


def verify_password(password, stored_hash):
    try:
        algorithm, n, r, p, salt_text, hash_text = stored_hash.split("$")
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(hash_text.encode("ascii"))
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def issue_session(user):
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    create_session(user["id"], token_hash, expires_at)
    return token, expires_at


def authenticate(email, password):
    user = get_user_by_email(normalize_email(email))
    if not user or not verify_password(password, user["password_hash"]):
        raise ValueError("Invalid email or password")
    return user


def user_response(user):
    return {"id": user["id"], "email": user["email"], "created_at": user["created_at"]}