# firebase_utils.py  — compact & safe

import os
from datetime import datetime
from typing import List, Dict, Optional

# Load .env if python-dotenv exists (won't crash if missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Try to import firebase; if missing, we'll fallback
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except Exception:
    firebase_admin = None
    credentials = None
    firestore = None


class FirebaseLogger:
    """Real Firestore logger (requires firebase-admin + env vars)."""
    def __init__(self) -> None:
        if not (firebase_admin and credentials and firestore):
            raise RuntimeError("firebase_admin not installed. Run: pip install firebase-admin python-dotenv")

        cred_path = os.getenv("FIREBASE_CREDENTIALS")
        project_id = os.getenv("FIREBASE_PROJECT_ID")

        if not cred_path or not os.path.exists(cred_path):
            raise RuntimeError("Set FIREBASE_CREDENTIALS to your service account JSON path (env or .env).")

        # Initialize app only once
        if not getattr(firebase_admin, "_apps", None):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {"projectId": project_id} if project_id else None)

        self.db = firestore.client()

    def log_prediction(self, image_path: str, rows: List[Dict], user: Optional[dict] = None) -> None:
        doc: Dict[str, object] = {
            "image_path": image_path,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "detections": rows,
        }
        if user:
            doc["user"] = {
                "uid": user.get("localId"),
                "email": user.get("email"),
            }
        self.db.collection("predictions").add(doc)
        print(f"[Firebase] pushed {len(rows)} detections for {os.path.basename(image_path)}")


class NullFirebaseLogger:
    """No-op fallback so pipeline keeps running without Firebase."""
    def log_prediction(self, image_path: str, rows: List[Dict], user: Optional[dict] = None) -> None:
        # Optional: print for visibility
        # print(f"[Firebase disabled] {len(rows)} detections for {os.path.basename(image_path)} user={user and user.get('email')}")
        pass


def get_firebase_logger():
    """Returns FirebaseLogger if configured; otherwise a no-op logger."""
    try:
        return FirebaseLogger()
    except Exception as e:
        print(f"[Firebase disabled] {e}")
        return NullFirebaseLogger()
