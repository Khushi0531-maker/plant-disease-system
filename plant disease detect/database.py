"""
User Management and Search History Database Module for FloraScan AI.
Powered by SQLite3 for persistent local storage, secure password hashing,
and personalized record keeping.
"""

import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "flora_users.db")


def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes database tables if they don't already exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Users Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            farm_type TEXT DEFAULT 'General Farming / Garden',
            location TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 2. Search History and Diagnostic Records Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            crop TEXT NOT NULL,
            disease TEXT NOT NULL,
            status TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            search_type TEXT NOT NULL, -- 'CNN Leaf Scan' or 'Crop Catalog Search'
            solution_summary TEXT,
            notes TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
        """)

        conn.commit()


def hash_password(password: str) -> str:
    """Hashes a password with salt using SHA-256."""
    salt = "florascan_agri_secure_salt_2026"
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def register_user(username: str, email: str, full_name: str, password: str, farm_type: str = "General Farming / Garden", location: str = ""):
    """
    Registers a new user.
    Returns: (success: bool, message: str, user_dict: dict | None)
    """
    username = username.strip().lower()
    email = email.strip().lower()
    full_name = full_name.strip()

    if not username or not email or not password or not full_name:
        return False, "Please fill in all required fields.", None

    if len(password) < 6:
        return False, "Password must be at least 6 characters long.", None

    p_hash = hash_password(password)

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, email, full_name, password_hash, farm_type, location)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (username, email, full_name, p_hash, farm_type, location))
            conn.commit()
            
            user_id = cursor.lastrowid
            user_data = {
                "id": user_id,
                "username": username,
                "email": email,
                "full_name": full_name,
                "farm_type": farm_type,
                "location": location
            }
            return True, "Registration successful! You can now log in.", user_data
    except sqlite3.IntegrityError as e:
        err_msg = str(e).lower()
        if "username" in err_msg:
            return False, "Username is already taken. Please choose another.", None
        elif "email" in err_msg:
            return False, "Email address is already registered. Please log in.", None
        return False, "Account already exists with this username or email.", None
    except Exception as e:
        return False, f"Registration error: {e}", None


def authenticate_user(username_or_email: str, password: str):
    """
    Authenticates a user against username or email.
    Returns: (success: bool, message: str, user_dict: dict | None)
    """
    identifier = username_or_email.strip().lower()
    p_hash = hash_password(password)

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, username, email, full_name, password_hash, farm_type, location, created_at
                FROM users
                WHERE (username = ? OR email = ?)
            """, (identifier, identifier))
            row = cursor.fetchone()

            if not row:
                return False, "User account not found. Please check your credentials or register.", None

            if row["password_hash"] != p_hash:
                return False, "Incorrect password. Please try again.", None

            user_data = {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"],
                "farm_type": row["farm_type"],
                "location": row["location"],
                "created_at": row["created_at"]
            }
            return True, "Login successful!", user_data
    except Exception as e:
        return False, f"Authentication error: {e}", None


def add_search_record(user_id: int, crop: str, disease: str, status: str, confidence: float, search_type: str, solution_summary: str, notes: str = ""):
    """Logs a search or diagnostic scan to the user's history."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO search_records (user_id, crop, disease, status, confidence, search_type, solution_summary, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, crop, disease, status, float(confidence), search_type, solution_summary, notes))
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        print(f"Error adding search record: {e}")
        return None


def get_user_records(user_id: int, limit: int = 100):
    """Retrieves all past records for a specific user, sorted newest first."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, crop, disease, status, confidence, search_type, solution_summary, notes, created_at
                FROM search_records
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        print(f"Error fetching user records: {e}")
        return []


def delete_record(record_id: int, user_id: int):
    """Deletes a specific record belonging to a user."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM search_records WHERE id = ? AND user_id = ?", (record_id, user_id))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error deleting record: {e}")
        return False


def clear_user_history(user_id: int):
    """Clears all search history for a user."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM search_records WHERE user_id = ?", (user_id,))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error clearing user history: {e}")
        return False


# Initialize DB on load
init_db()
