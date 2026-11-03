import hashlib
import hmac
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import jwt

from app.core import config
from app.core.db import get_conn

ITERATIONS = 200_000


class AuthError(Exception):
    pass


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS).hex()


def register(username: str, password: str):
    salt = os.urandom(16)
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, _hash(password, salt), salt.hex(), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise AuthError("username already taken")
    finally:
        conn.close()


def authenticate(username: str, password: str) -> str:
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    # same error for unknown user and wrong password
    if row is None or not hmac.compare_digest(row["password_hash"], _hash(password, bytes.fromhex(row["salt"]))):
        raise AuthError("invalid username or password")
    return create_token(username)


def create_token(username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=config.JWT_EXPIRE_MINUTES)
    return jwt.encode({"sub": username, "exp": exp}, config.JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise AuthError("invalid or expired token")
    return payload["sub"]
