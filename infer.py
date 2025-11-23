""""
import argparse
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from PIL import Image

try:
    from firebase_utils import FirebaseLogger
except Exception:
    FirebaseLogger = None

def run_inference(weights, source, save=False, to_csv=None, push_firebase=False, conf=0.25):
    model = YOLO(weights)
    source_path = Path(source)

    if source_path.is_dir():
        images = [p for p in source_path.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.tif','.tiff']]
    else:
        images = [source_path]

    rows = []
    fb = FirebaseLogger() if (push_firebase and FirebaseLogger is not None) else None

    for img_path in images:
        res = model.predict(source=str(img_path), conf=conf, save=save, project='inference_out', name='pred', exist_ok=True, verbose=False)
        r = res[-1]
        classes = r.names
        for b in r.boxes:
            cls_id = int(b.cls.item())
            cls_nm = classes[cls_id]
            confv = float(b.conf.item())
            x1,y1,x2,y2 = [float(x) for x in b.xyxy[0].tolist()]
            rows.append({
                'image': str(img_path),
                'class_id': cls_id,
                'class_name': cls_nm,
                'confidence': confv,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })

        if fb:
            try:
                fb.log_prediction(str(img_path), rows=[r for r in rows if r['image']==str(img_path)])
            except Exception as e:
                print(f'Firebase error for {img_path.name}: {e}')

    if to_csv:
        pd.DataFrame(rows).to_csv(to_csv, index=False)
        print(f'Saved CSV: {to_csv}')
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='path to .pt weights')
    ap.add_argument('--source', required=True, help='image or folder')
    ap.add_argument('--save', action='store_true', help='save annotated outputs')
    ap.add_argument('--to_csv', type=str, default=None)
    ap.add_argument('--push_firebase', action='store_true')
    ap.add_argument('--conf', type=float, default=0.25)
    args = ap.parse_args()
    run_inference(**vars(args)) """

#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
from ultralytics import YOLO

# --- Firebase import (safe fallback) ---
try:
    # prefer helper that won’t crash when not configured
    from firebase_utils import get_firebase_logger  # type: ignore
    _fb_helper_available = True
except Exception:
    _fb_helper_available = False
    get_firebase_logger = None  # type: ignore


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_default_weights() -> Path | None:
    candidates = [
        Path("best.pt"),
        Path("best (2).pt"),
        Path("runs/detect/train/weights/best.pt"),
        Path("weights/best.pt"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_default_source() -> Path | None:
    for p in [Path("processed_images"), Path("processed"), Path("raw_images"), Path("raw")]:
        if p.exists():
            return p
    return None


def list_images(root: Path, recursive: bool) -> List[Path]:
    if root.is_file() and root.suffix.lower() in IMG_EXTS:
        return [root]
    if not root.is_dir():
        return []
    if recursive:
        return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def run_inference(
    weights: Path,
    source: Path,
    save: bool = False,
    to_csv: str | None = None,
    push_firebase: bool = False,
    conf: float = 0.25,
    recursive: bool = False,
) -> List[Dict]:
    """
    Run YOLOv8 detections on a single image or a folder (optionally recursive).
    Returns a list of dict rows (one per detection).
    """
    model = YOLO(str(weights))

    # Collect images
    images = list_images(source, recursive=recursive)
    if not images:
        print(f"[!] No images found under: {source}")
        return []

    # Firebase logger (no-op if not configured/available)
    fb = None
    if push_firebase and _fb_helper_available and callable(get_firebase_logger):  # type: ignore
        try:
            fb = get_firebase_logger()
        except Exception as e:
            print(f"[Firebase disabled] {e}")

    rows: List[Dict] = []
    for img_path in images:
        res = model.predict(
            source=str(img_path),
            conf=float(conf),
            save=save,
            project="inference_out",
            name="pred",
            exist_ok=True,
            verbose=False,
        )

        # Ultralytics returns a list; take the last result for this call
        r = res[-1]
        classes = r.names if hasattr(r, "names") else {}

        # If no boxes (e.g., classifier run), still handle gracefully
        found = 0
        for b in getattr(r, "boxes", []):
            cls_id = int(b.cls.item())
            cls_nm = classes.get(cls_id, str(cls_id)) if isinstance(classes, dict) else str(cls_id)
            confv = float(b.conf.item())
            x1, y1, x2, y2 = [float(x) for x in b.xyxy[0].tolist()]
            rows.append(
                {
                    "image": str(img_path),
                    "class_id": cls_id,
                    "class_name": cls_nm,
                    "confidence": confv,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            found += 1

        if found == 0 and hasattr(r, "probs") and r.probs is not None:
            # Classification case (top-1 only)
            import numpy as np

            vec = getattr(r.probs, "data", r.probs)
            if hasattr(vec, "detach"):
                vec = vec.detach()
            if hasattr(vec, "cpu"):
                vec = vec.cpu()
            vec = np.array(vec)
            if vec.ndim == 2 and vec.shape[0] == 1:
                vec = vec[0]
            top_id = int(vec.argmax())
            top_conf = float(vec[top_id])
            top_name = classes.get(top_id, str(top_id)) if isinstance(classes, dict) else str(top_id)
            rows.append(
                {
                    "image": str(img_path),
                    "class_id": top_id,
                    "class_name": top_name,
                    "confidence": top_conf,
                    "x1": None,
                    "y1": None,
                    "x2": None,
                    "y2": None,
                }
            )
            found = 1

        # Per-image console summary (nice when running batches)
        if found:
            # Count by class
            by_cls: Dict[str, int] = {}
            for rr in [d for d in rows if d["image"] == str(img_path)]:
                nm = rr["class_name"]
                by_cls[nm] = by_cls.get(nm, 0) + 1
            counts = ", ".join(f"{k}:{v}" for k, v in by_cls.items())
            print(f"[✓] {img_path.name}: {found} detections  ({counts})")
        else:
            print(f"[–] {img_path.name}: no detections")

        # Optional Firebase log for THIS image only
        if fb:
            try:
                per_image = [d for d in rows if d["image"] == str(img_path)]
                fb.log_prediction(str(img_path), rows=per_image, user=None)
            except Exception as e:
                print(f"[Firebase] error for {img_path.name}: {e}")

    # Save CSV if requested
    if to_csv:
        df = pd.DataFrame(rows)
        df.to_csv(to_csv, index=False)
        print(f"[CSV] Saved: {to_csv}")

    return rows


def main():
    # Resolve helpful defaults so you can just press Run
    default_w = find_default_weights()
    default_s = find_default_source()

    ap = argparse.ArgumentParser(description="Run corn maturity detection with YOLOv8.")
    ap.add_argument("--weights", type=str, default=str(default_w) if default_w else None,
                    help="Path to .pt model weights (default: auto-detect best.pt)")
    ap.add_argument("--source", type=str, default=str(default_s) if default_s else None,
                    help="Image file or folder (default: auto-detect processed_images/)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    ap.add_argument("--save", action="store_true", help="Save annotated images to inference_out/pred/")
    ap.add_argument("--to_csv", type=str, default=None, help="Save detections to CSV")
    ap.add_argument("--push_firebase", action="store_true", help="Log detections to Firestore")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders when source is a folder")

    args = ap.parse_args()

    # Friendly messages about defaults
    if args.weights is None:
        print("[!] No --weights provided and none auto-detected. Please pass --weights /path/to/best.pt")
        return
    if args.source is None:
        print("[!] No --source provided and none auto-detected. Please pass --source /path/to/images_or_image")
        return

    weights = Path(args.weights).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()

    print(f"Model:   {weights}")
    print(f"Source:  {source}")
    print(f"Conf:    {args.conf} | Save annotated: {args.save} | Firebase: {args.push_firebase} | Recursive: {args.recursive}")

    rows = run_inference(
        weights=weights,
        source=source,
        conf=args.conf,
        save=args.save,
        to_csv=args.to_csv,
        push_firebase=args.push_firebase,
        recursive=args.recursive,
    )
    print(f"\nDone. Total detections: {len(rows)}")
    if not args.save:
        print("💡 Tip: add --save to write annotated images to inference_out/pred/")


if __name__ == "__main__":
    main()
