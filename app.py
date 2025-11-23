# ===============================================================
# AeroCorn — Corn Maturity Detector + Image Preprocessor (EN/BM)
# Clean version with top header navigation + account settings
# ===============================================================
from pathlib import Path
from typing import Tuple
import base64
import os
import requests
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ---------------- Page setup ----------------
st.set_page_config(page_title="🌽 AeroCorn — Corn Maturity AI", layout="wide")


# ---------------- Helpers ----------------
def get_image_base64(path: str) -> str:
    """Read image and return base64 string (safe fallback if missing)."""
    p = Path(path)
    if not p.exists():
        st.warning(f"Logo not found at: {p}")
        return ""
    with p.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------- Top Navigation Bar (sticky) ----------------
def set_mode(m: str):
    st.session_state["mode"] = m

def top_navbar(logo_b64: str, email: str = ""):
    """Modern top navbar with logo + title + subtitle + navigation buttons."""
    st.session_state.setdefault("mode", "Detector")

    st.markdown("""
    <style>
      /* Tight top bar, remove Streamlit default padding */
      .block-container { padding-top: 0rem !important; }
      .topbar {
          position: sticky;
          top: 0;
          z-index: 9999;
          display: flex;
          align-items: center;
          justify-content: space-between;
          background: #F4FBF4;
          border-bottom: 1px solid #E2F2E4;
          padding: 0.5rem 1.2rem;
          margin: 0 -1rem 1rem -1rem;
      }
      .brand {
          display: flex;
          align-items: center;
          gap: 0.8rem;
      }
      .brand img {
          width: 42px;
          height: 42px;
          border-radius: 10px;
      }
      .brand-title {
          font-size: 1.4rem;
          font-weight: 800;
          color: #236C2A;
          margin-bottom: -4px;
      }
      .brand-sub {
          font-size: 0.8rem;
          color: #335C3A;
          opacity: 0.85;
      }
      .nav-buttons {
          display: flex;
          gap: 0.6rem;
      }
      .nav-buttons button {
          border-radius: 12px !important;
          font-weight: 700 !important;
      }
    </style>
    """, unsafe_allow_html=True)

    # --- top bar layout ---
    with st.container():
        st.markdown('<div class="topbar">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 3, 3])

        # Left: Logo + Titles
        with col1:
            st.markdown(
                f"""
                <div class="brand">
                    <img src="data:image/png;base64,{logo_b64}" alt="AeroCorn Logo">
                    <div>
                        <div class="brand-title">AeroCorn</div>
                        <div class="brand-sub">AI-Powered Corn Harvest Detection</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Middle: Nav buttons
        with col2:
            mode = st.session_state["mode"]
            b1, b2, b3 = st.columns(3)
            with b1:
                st.button("🌽 Detector", type=("primary" if mode == "Detector" else "secondary"),
                          use_container_width=True, on_click=lambda: st.session_state.update(mode="Detector"))
            with b2:
                st.button("🖼️ Preprocess", type=("primary" if mode == "Preprocess" else "secondary"),
                          use_container_width=True, on_click=lambda: st.session_state.update(mode="Preprocess"))
            with b3:
                st.button("👤 Account", type=("primary" if mode == "Account" else "secondary"),
                          use_container_width=True, on_click=lambda: st.session_state.update(mode="Account"))

        # Right: user email (plain text)
        with col3:
            if email:
                st.markdown(
                    f"<div style='text-align:right; color:#335C3A; font-weight:600; font-size:0.9rem;'>{email}</div>",
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Firebase (Pyrebase client) ----------------
try:
    import pyrebase  # pip install pyrebase4
except Exception:
    st.error("Missing pyrebase4. Run: pip install pyrebase4")
    st.stop()

if "firebase" not in st.secrets:
    st.error("Missing .streamlit/secrets.toml with [firebase] config.")
    st.stop()

cfg = dict(st.secrets["firebase"])
cfg.setdefault("databaseURL", f"https://{cfg['projectId']}.firebaseio.com")
# standardize bucket
if cfg.get("storageBucket", "").endswith("firebasestorage.app"):
    cfg["storageBucket"] = f"{cfg['projectId']}.appspot.com"

firebase = pyrebase.initialize_app(cfg)
auth = firebase.auth()
API_KEY = cfg.get("apiKey", "")

# session holder for signed-in user (pyrebase user dict)
st.session_state.setdefault("user", None)


# ---- REST helpers for account management (Pyrebase lacks admin methods)
def fb_change_password(id_token: str, new_password: str):
    """Change password using Firebase REST Identity Toolkit."""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:update?key={API_KEY}"
    payload = {"idToken": id_token, "password": new_password, "returnSecureToken": True}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def fb_delete_account(id_token: str):
    """Delete account using Firebase REST Identity Toolkit."""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:delete?key={API_KEY}"
    payload = {"idToken": id_token}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------- Green theme (colour-only edits) ----------------
st.markdown(
    """
<style>
/* ========= Soft Green Theme ========= */
.auth-card{
  max-width:680px; margin:4rem auto 0; padding:2rem; background:#FFFFFF;
  border-radius:16px; text-align:center; box-shadow:0 10px 28px rgba(0,0,0,.06);
  border:1px solid #E7F3E8;
}
.auth-card img{ width:120px; height:120px; border-radius:14px; }
.auth-title{ font-size:1.8rem; font-weight:800; color:#236C2A; }
.auth-sub{ color:#335C3A; opacity:.95; }

html, body, .stApp{ font-family:'Poppins','Helvetica',sans-serif; background:#F4FBF4 !important; color:#0B2E13 !important; }
header[data-testid="stHeader"]{ background:#F4FBF4 !important; border-bottom:1px solid #E2F2E4; }
header[data-testid="stHeader"] *{ color:#0B2E13 !important; }

section[data-testid="stSidebar"]{ background:#EAF7EE !important; border-right:1px solid #D7ECDD !important; }
section[data-testid="stSidebar"] *{ color:#0B2E13 !important; }

.sidebar-logo{ display:flex; flex-direction:column; align-items:center; padding:1rem 0 0.8rem; }
.sidebar-logo img{ width:140px; height:140px; border-radius:16px; background:#FFFFFF; box-shadow:0 6px 18px rgba(0,0,0,.06); }
.sidebar-logo h1{ margin:.9rem 0 .2rem 0; font-size:2rem; line-height:1.1; font-weight:800; color:#236C2A; text-align:center; }
.sidebar-hr{ margin:10px 0 14px 0; border:0; height:1px; background:#D7ECDD; }

.sidebar a { color:#1E88E5 !important; text-decoration: none; }
.sidebar a:hover { text-decoration: underline; }

.stButton>button{
  width:100%; height:2.8em; border-radius:12px;
  background:#FFFFFF !important; color:#236C2A !important;
  border:1px solid #AFCFBA !important; font-weight:700;
  transition:all .15s ease;
}
.stButton>button:hover{ background:#2E7D32 !important; color:#FFFFFF !important; border-color:#1E5A23 !important; }

input[type="radio"], input[type="checkbox"]{ accent-color:#2E7D32; }
h1,h2,h3,.stMarkdown h1,.stMarkdown h2{ color:#0B2E13 !important; font-weight:800 !important; }

div[data-testid="stExpander"]{ border:1px solid #D7ECDD !important; border-radius:10px !important; overflow:hidden; }
div[data-testid="stExpander"] summary{ background:#E8F5E9 !important; color:#0B2E13 !important; font-weight:700; padding:.6rem 1rem; }
div[data-testid="stExpander"] div[role="region"]{ background:#F9FEFA !important; color:#0B2E13 !important; padding:1rem; }

.stFileUploader>div{ background:#FFFFFF !important; border:1px solid #D7ECDD !important; border-radius:10px; }
input, textarea, select{ background:#FFFFFF !important; color:#0B2E13 !important; border:1px solid #CFE6D2 !important; border-radius:8px !important; }
input::placeholder, textarea::placeholder{ color:#6E8F74 !important; }
.stDataFrame, .stDataFrame *{ color:#0B2E13 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- Auth helpers ----------------
def is_authed() -> bool:
    u = st.session_state.get("user")
    return isinstance(u, dict) and bool(u.get("idToken"))


def render_login(logo_b64: str):
    st.markdown(
        f"""
        <div class="auth-card">
            <img src="data:image/png;base64,{logo_b64}" alt="AeroCorn Logo">
            <div class="auth-title">AeroCorn</div>
            <div class="auth-sub">Smart Corn Maturity AI System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_login, tab_signup = st.tabs(["Login", "Create account"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        c1, c2 = st.columns(2)
        if c1.button("Login", key="btn_login"):
            try:
                user = auth.sign_in_with_email_and_password(email.strip(), password)
                st.session_state.user = user
                st.rerun()
            except Exception:
                st.error("Invalid email or password.")
        if c2.button("Send password reset", key="btn_reset"):
            try:
                auth.send_password_reset_email(email.strip())
                st.info("Password reset email sent.")
            except Exception:
                st.warning("Enter a valid email to reset.")

    with tab_signup:
        se = st.text_input("New email", key="signup_email")
        sp = st.text_input("New password", type="password", key="signup_pass")
        if st.button("Create account", key="btn_signup"):
            try:
                auth.create_user_with_email_and_password(se.strip(), sp)
                st.success("Account created! Please log in.")
            except Exception:
                st.error("Sign up failed. Try a stronger password or different email.")


def render_sidebar_branding(logo_b64: str):
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-logo">
                <img src="data:image/png;base64,{logo_b64}" alt="AeroCorn Logo">
                <h1>AeroCorn</h1>
            </div>
            <hr class="sidebar-hr">
            """,
            unsafe_allow_html=True,
        )
        _user = st.session_state.get("user") or {}
        st.write(f"👋 [{_user.get('email','User')}]({_user.get('email','')})")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()


# ---------------- Account Settings (uses REST helpers) ----------------
def account_settings():
    st.subheader("👤 Account Settings")

    user = st.session_state.get("user")
    if not user:
        st.warning("You must be logged in to modify your account.")
        return

    # Change password
    st.write("### Change Password")
    new_pass = st.text_input("Enter new password", type="password")
    if st.button("Update Password"):
        try:
            fb_change_password(user["idToken"], new_pass)
            st.success("Password updated successfully.")
        except Exception as e:
            st.error(f"Error updating password: {e}")

    st.write("---")

    # Delete account
    st.write("### Delete Account")
    if st.button("Delete My Account"):
        try:
            fb_delete_account(user["idToken"])
            st.warning("Account deleted. Please close this window.")
            st.session_state.user = None
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting account: {e}")


# ---------------- Language toggle ----------------
lang = st.sidebar.radio("Language / Bahasa", ["EN", "BM"])
def T(en, bm): return en if lang == "EN" else bm


# ---------------- Model utilities ----------------
@st.cache_resource
def load_model(weights_path: str):
    p = Path(weights_path).expanduser().resolve()
    if not p.exists():
        st.error(f"Model file not found: {p}")
        st.stop()
    return YOLO(str(p))

# --- add these new helpers right below the function above ---
@st.cache_resource
def load_det_model():
    """Stage-1 detection model (corn localization)."""
    return YOLO(str(Path("weight/bestS1.pt").resolve()))

@st.cache_resource
def load_cls_model():
    """Stage-2 maturity classifier model."""
    return YOLO(str(Path("weight/aerocorn_maturity_classifier.pt").resolve()))

def recommend(detections: pd.DataFrame, threshold=0.5) -> Tuple[str, str]:
    if detections.empty:
        return T("No corn detected", "Tiada tanaman dikesan"), "⚠️"
    if "rank" in detections.columns:  # classification
        top1 = detections.sort_values("rank").iloc[0]
        is_mature = (
            str(top1["class_name"]).strip().lower() == MATURE_LABEL
            and float(top1["confidence"]) >= float(threshold)
        )
        return (
            (T("Ready for Harvest", "Sedia untuk Tuai"), "✅")
            if is_mature
            else (T("Still Immature", "Masih Belum Matang"), "🌱")
        )
    # detection
    any_mature = (
        detections["class_name"].str.strip().str.lower().eq(MATURE_LABEL)
        & (detections["confidence"] >= float(threshold))
    ).any()
    return (
        (T("Ready for Harvest", "Sedia untuk Tuai"), "✅")
        if any_mature
        else (T("Still Immature", "Masih Belum Matang"), "🌱")
    )


def yolo_predict(model, image: Image.Image, conf=0.25):
    """Return (DataFrame, annotated PIL image). Supports YOLO detect & classify."""
    res = model.predict(source=image, conf=float(conf), save=False, verbose=False)
    r = res[0]
    rows = []

    # classification
    if hasattr(r, "probs") and r.probs is not None:
        names = r.names if hasattr(r, "names") else {}
        import numpy as np
        vec = getattr(r.probs, "data", r.probs)
        if hasattr(vec, "detach"):
            vec = vec.detach()
        if hasattr(vec, "cpu"):
            vec = vec.cpu()
        vec = np.array(vec)
        if vec.ndim == 2 and vec.shape[0] == 1:
            vec = vec[0]
        idx = vec.argsort()[::-1][:5].tolist()
        for rank, cid in enumerate(idx, start=1):
            rows.append(
                {
                    "rank": rank,
                    "class_id": cid,
                    "class_name": names.get(cid, str(cid))
                    if isinstance(names, dict)
                    else str(cid),
                    "confidence": float(vec[cid]),
                }
            )
        annotated = r.plot()[:, :, ::-1]
        if rows:
            st.caption(f"Top-1: {rows[0]['class_name']} ({rows[0]['confidence']:.2f})")
        return pd.DataFrame(rows), Image.fromarray(annotated)

    # detection
    nboxes = len(getattr(r, "boxes", []))
    for b in getattr(r, "boxes", []):
        cls_id = int(b.cls.item())
        cls_nm = r.names.get(cls_id, str(cls_id)) if hasattr(r, "names") else str(cls_id)
        confv = float(b.conf.item())
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        rows.append(
            {
                "class_id": cls_id,
                "class_name": cls_nm,
                "confidence": confv,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    annotated = r.plot()[:, :, ::-1]
    st.caption(f"{T('Detected boxes', 'Kotak Dikesan')}: {nboxes}")
    return pd.DataFrame(rows), Image.fromarray(annotated)

def yolo_detect_then_classify(det_model, cls_model, image: Image.Image,
                              det_conf=0.25, min_area=300,
                              det_imgsz=640, cls_imgsz=320):
    # ---- Stage 1: detection ----
    det_kwargs = dict(conf=float(det_conf), save=False, verbose=False)
    if det_imgsz:
        det_kwargs["imgsz"] = int(det_imgsz)
    det_res = det_model.predict(source=image, **det_kwargs)[0]
    boxes = getattr(det_res, "boxes", None)

    # base annotated image from detector (or original if plotting fails)
    try:
        ann_img = Image.fromarray(det_res.plot()[:, :, ::-1])
    except Exception:
        ann_img = image.copy()

    rows = []

    # ======= No boxes → treat as "not corn" (whole-image classify; no coords) =======
    if not boxes or len(boxes) == 0:
        cls_res = cls_model.predict(source=image, imgsz=int(cls_imgsz), save=False, verbose=False)[0]
        top_name, top_conf = "unknown", 0.0
        if hasattr(cls_res, "probs") and cls_res.probs is not None:
            import numpy as np
            names = getattr(cls_res, "names", {})
            vec = getattr(cls_res.probs, "data", cls_res.probs)
            if hasattr(vec, "detach"): vec = vec.detach()
            if hasattr(vec, "cpu"):    vec = vec.cpu()
            vec = np.array(vec)
            if vec.ndim == 2 and vec.shape[0] == 1:
                vec = vec[0]
            top_idx = int(vec.argmax())
            top_conf = float(vec[top_idx])
            top_name = names[top_idx] if isinstance(names, dict) and top_idx in names else str(top_idx)

        # keep columns your downstream code expects (class_name/confidence)
        rows.append({
            "box_id": None,
            "det_conf": None,
            "class_id": -1,
            "class_name": str(top_name),       # status
            "confidence": float(top_conf),     # status_conf
            "x1": None, "y1": None, "x2": None, "y2": None
        })
        return pd.DataFrame(rows), ann_img

    # ======= Per-image overlay styling (consistent) =======
    try:
        FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)  # fixed = consistent
    except Exception:
        FONT = ImageFont.load_default()
    PAD, OUTLINE = 8, 3
    draw = ImageDraw.Draw(ann_img)

    # ======= Stage 2: per-crop classification =======
    for i, b in enumerate(boxes, start=1):
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        area = (x2 - x1) * (y2 - y1)
        if area < float(min_area):
            continue

        det_box_conf = float(b.conf.item())  # detector confidence for this corn box

        crop = image.crop((x1, y1, x2, y2))
        cls_res = cls_model.predict(source=crop, imgsz=int(cls_imgsz), save=False, verbose=False)[0]

        # top-1 class from classifier
        top_name, top_conf = "unknown", 0.0
        if hasattr(cls_res, "probs") and cls_res.probs is not None:
            import numpy as np
            names = getattr(cls_res, "names", {})
            vec = getattr(cls_res.probs, "data", cls_res.probs)
            if hasattr(vec, "detach"): vec = vec.detach()
            if hasattr(vec, "cpu"):    vec = vec.cpu()
            vec = np.array(vec)
            vec = vec[0] if (vec.ndim == 2 and vec.shape[0] == 1) else vec
            top_idx = int(vec.argmax())
            top_conf = float(vec[top_idx])
            top_name = names[top_idx] if isinstance(names, dict) and top_idx in names else str(top_idx)

        # OPTIONAL overlay (big label on each box)
            pass

        # keep original columns + add clean fields for your table
        rows.append({
            "box_id": i,                        # clean index for the corn box
            "det_conf": det_box_conf,           # detector confidence
            "class_id": -1,
            "class_name": str(top_name),        # status (mature/immature)
            "confidence": float(top_conf),      # status_conf
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

    return pd.DataFrame(rows), ann_img

# ---------------- Detector page ----------------
def render_detector():
    st.header("🌽 Corn Maturity Detector")
    st.caption(
        T(
            "Upload your corn images below and let AI estimate if they are ready for harvest.",
            "Muat naik gambar jagung anda di bawah dan biarkan AI menilai sama ada sedia untuk dituai.",
        )
    )

    # --- Hardcoded settings, no UI ---
    # --- Models + thresholds ---
    det_conf = 0.25
    rec_thresh = 0.5
    log_to_firebase = False

    # load BOTH models (cached)
    det_model = load_det_model()
    cls_model = load_cls_model()

    # resolve mature label from model classes
    def resolve_mature_label(m) -> str:
        try:
            names = getattr(m.model, "names", getattr(m, "names", {}))
            lowered = (
                {i: str(v).strip().lower() for i, v in names.items()}
                if isinstance(names, dict)
                else {i: str(v).strip().lower() for i, v in enumerate(names)}
            )
            for v in lowered.values():
                if v in {"mature", "ripe", "corn-ripe", "ripe_corn", "corn_ripe"}:
                    return v
        except Exception:
            pass
        return "mature"

    global MATURE_LABEL
    MATURE_LABEL = resolve_mature_label(cls_model)

    uploads = st.file_uploader(
        T("📸 Upload corn image(s)", "📸 Muat naik gambar jagung"),
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploads:
        st.info(
            T(
                "Upload one or more images to begin.",
                "Muat naik satu atau lebih gambar untuk bermula.",
            )
        )
        return

    # Optional: Firebase logger class (if you created it)
    try:
        from firebase_utils import FirebaseLogger  # your own helper
        fb_logger = FirebaseLogger() if log_to_firebase else None
    except Exception:
        fb_logger = None

    det_all = []
    for up in uploads:
        image = Image.open(up).convert("RGB")
        with st.spinner(f"{T('Processing', 'Memproses')} {up.name}..."):
            det_df, annotated = yolo_detect_then_classify(det_model, cls_model, image,
                                                          det_conf=det_conf, min_area=300)

        # ✅ Hard guard: if there are no detection boxes → say "This is not corn"
        if det_df.empty or ("x1" in det_df and det_df["x1"].isna().all()):
            rec_txt, rec_icon = T("This is not corn", "Ini bukan jagung"), "❌"
        else:
            rec_txt, rec_icon = recommend(det_df, threshold=rec_thresh)


        with st.container(border=True):
            st.subheader(f"📷 {up.name}")
            c1, c2 = st.columns([3, 2])
            with c1:
                st.image(annotated, caption=f"{rec_icon} {rec_txt}", use_container_width=True)
                # Count mature vs immature
                # Only show maturity summary if there are real boxes (not "This is not corn")
                if not det_df.empty and not ("x1" in det_df and det_df["x1"].isna().all()):
                    mature_count = int((det_df["class_name"].str.lower() == "mature").sum())
                    immature_count = int((det_df["class_name"].str.lower() == "immature").sum())

                    summary_text = (
                        f"✅ {mature_count} Mature (Ready for Harvest)\n"
                        f"🌱 {immature_count} Immature (Still Growing)"
                    )

                    st.markdown(
                        f"""
                        <div style='
                            background-color:#F9FEFA;
                            border:1px solid #D7ECDD;
                            border-radius:10px;
                            padding:10px 15px;
                            font-size:1rem;
                            font-weight:600;
                            color:#0B2E13;
                            line-height:1.6;
                            '>
                            {summary_text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with c2:
                st.markdown(f"#### 🌾 {T('Summary', 'Ringkasan')}")
                st.info(f"{T('Prediction', 'Ramalan')}: **{rec_txt}** {rec_icon}")

                # 🔎 Technical table for evaluator (clear & simple)
                if rec_txt.strip().lower() not in ["this is not corn", "ini bukan jagung"]:
                    df = det_df.copy()

                    # Ensure clean identifiers even if yolo_detect_then_classify didn't supply them
                    if "box_id" not in df.columns:
                        # Only assign IDs if we have coordinates (i.e., real boxes)
                        if {"x1", "y1", "x2", "y2"}.issubset(df.columns) and not df["x1"].isna().all():
                            df.insert(0, "box_id", range(1, len(df) + 1))
                        else:
                            df["box_id"] = None

                    # Detector confidence may not be present; show "—" if missing
                    if "det_conf" not in df.columns:
                        df["det_conf"] = None

                    # Build the clean view
                    view_cols = ["box_id", "det_conf", "class_name", "confidence"]
                    view_cols = [c for c in view_cols if c in df.columns]
                    display_df = df[view_cols].rename(columns={
                        "box_id": "corn_box",
                        "det_conf": "det_conf",
                        "class_name": "status",
                        "confidence": "status_conf",
                    }).copy()

                    # Format confidences to % strings
                    if "det_conf" in display_df.columns:
                        display_df["det_conf"] = display_df["det_conf"].apply(
                            lambda v: (f"{float(v) * 100:.0f}%" if pd.notnull(v) else "—")
                        )
                    if "status_conf" in display_df.columns:
                        display_df["status_conf"] = display_df["status_conf"].apply(
                            lambda v: f"{float(v) * 100:.0f}%"
                        )

                    # Show counts (same as before)
                    n_mature = (df.get("class_name", "").str.lower() == "mature").sum()
                    n_immature = (df.get("class_name", "").str.lower() == "immature").sum()
                    st.success(f"✅ {n_mature} {T('Mature (Ready for Harvest)', 'Matang (Sedia Dituai)')}")
                    st.warning(f"🌱 {n_immature} {T('Immature (Still Growing)', 'Belum Matang (Masih Tumbuh)')}")

                    # Render the concise table
                    st.dataframe(display_df, use_container_width=True, height=260)

                if fb_logger:
                    try:
                        fb_logger.log_prediction(
                            up.name, det_df.to_dict("records"), st.session_state.user
                        )
                        st.success("✅ Logged to Firebase")
                    except Exception as e:
                        st.warning(f"Firebase log failed: {e}")
                st.markdown(
                    "💡 "
                    + T(
                        "Tip: Harvest when kernels are deep yellow and dry.",
                        "Tip: Tuai apabila biji berwarna kuning pekat dan kering.",
                    )
                )
        det_all.append(det_df.assign(image=up.name))

    # ---- Combined detections (show only when real boxes exist) ----
    if det_all:
        combined = pd.concat(det_all, ignore_index=True)

        # Buang baris 'not corn' (tiada koordinat)
        need_cols = {"x1", "y1", "x2", "y2"}
        if need_cols.issubset(combined.columns):
            combined = combined.dropna(subset=list(need_cols))

        # Jika tiada apa-apa yang sah, jangan tunjuk seksyen ini langsung
        if len(combined) > 0:
            # Opsyenal: kemas kolum & format confidence (%)
            if "confidence" in combined.columns and "det_conf" not in combined.columns:
                combined["det_conf"] = (combined["confidence"] * 100).round(1).astype(str) + "%"

            # Susunan kolum mesra pembaca (ikut apa yang wujud)
            order = [c for c in ["image", "class_name", "det_conf", "x1", "y1", "x2", "y2", "class_id"] if
                     c in combined.columns]
            if order:
                combined = combined[order]

            with st.expander("🧾 Combined Detections (advanced)", expanded=False):
                st.dataframe(combined, use_container_width=True)
        # else: tiada kotak sah → senyap, tak tunjuk apa-apa

    with st.expander("ℹ️ About"):
        st.markdown(
            T(
                """
**Corn Maturity Detector v1.0**

- Detects corn ripeness using YOLOv8 AI model  
- Works best with clear daylight images  
- Designed for farmers and agriculture officers 🌾
""",
                """
**Pengesan Kematangan Jagung v1.0**

- Mengesan kematangan jagung menggunakan model AI YOLOv8  
- Berfungsi terbaik dengan gambar jelas pada waktu siang  
- Direka khas untuk petani dan pegawai pertanian 🌾
""",
            )
        )


# ---------------- Preprocessor page ----------------
def enhance_and_resize_pil(img_pil, size=640):
    try:
        import cv2
        import numpy as np

        img = np.array(img_pil.convert("RGB"))[:, :, ::-1]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        out = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        out = cv2.resize(out, (int(size), int(size)))
        out = out[:, :, ::-1]
        return Image.fromarray(out)
    except Exception:
        from PIL import ImageOps, ImageEnhance

        img = ImageOps.autocontrast(img_pil.convert("RGB"))
        img = ImageEnhance.Color(img).enhance(1.05)
        return img.resize((int(size), int(size)))


def render_preprocessor():
    st.header("🖼️ Image Preprocessor")
    st.caption(
        T(
            "Enhance contrast (CLAHE) and resize to uniform square, then download as ZIP.",
            "Tingkatkan kontras (CLAHE) dan saiz semula segi empat sama, kemudian muat turun sebagai ZIP.",
        )
    )

    size = st.number_input(T("Output size (px)", "Saiz output (px)"), 128, 2048, 640, 32)
    up_imgs = st.file_uploader(
        T("Upload raw image(s)", "Muat naik imej mentah"),
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if not up_imgs:
        st.info(T("Upload a few images to begin.", "Muat naik beberapa imej untuk bermula."))
        return

    from io import BytesIO
    import zipfile

    thumbs, zip_buffer = [], BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in up_imgs:
            try:
                pil_in = Image.open(f).convert("RGB")
                pil_out = enhance_and_resize_pil(pil_in, size=size)
                if len(thumbs) < 12:
                    thumbs.append(pil_out.copy())
                bio = BytesIO()
                pil_out.save(bio, format="JPEG", quality=92)
                zf.writestr(Path(f.name).with_suffix(".jpg").name, bio.getvalue())
            except Exception as e:
                st.warning(f"⚠️ {f.name}: {e}")

    st.markdown("**Preview**")
    cols = st.columns(4)
    for i, im in enumerate(thumbs):
        with cols[i % 4]:
            st.image(im, use_container_width=True)

    zip_buffer.seek(0)
    st.download_button(
        label=T("⬇️ Download processed ZIP", "⬇️ Muat turun ZIP terproses"),
        data=zip_buffer,
        file_name="processed_images.zip",
        mime="application/zip",
    )


# ---------------- Gate + layout ----------------
# 1) Auth gate
logo_b64 = get_image_base64("aerocorn_logo.png")
if not is_authed():
    render_login(logo_b64)
    st.stop()

# 2) Sidebar branding
render_sidebar_branding(logo_b64)

# 3) Top header nav
_top_user = st.session_state.get("user") or {}
top_navbar(logo_b64, email=_top_user.get("email", ""))

# 4) Page switcher
mode = st.session_state.get("mode", "Detector")
if mode == "Detector":
    render_detector()
elif mode == "Preprocess":
    render_preprocessor()
elif mode == "Account":
    account_settings()
