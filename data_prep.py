"""
data_prep.py — Enhance and resize corn images for training/inference.

✅ Two ways to run:
1️⃣  CLI mode:
      python data_prep.py --src raw_images --dst processed_images --size 640
2️⃣  GUI mode (auto or manual):
      python data_prep.py
      or
      streamlit run data_prep.py
"""


import argparse
import os
import glob
import cv2
import sys


# ---------------- IMAGE ENHANCEMENT ----------------
def enhance_image(img):
    """Enhance image brightness and contrast using CLAHE."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


# ---------------- FOLDER PROCESSOR ----------------
def process_folder(src, dst, size=640):
    """Process all images in a folder: enhance + resize + save."""
    os.makedirs(dst, exist_ok=True)
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    count = 0

    for p in glob.glob(os.path.join(src, '**', '*'), recursive=True):
        if p.lower().endswith(exts):
            img = cv2.imread(p)
            if img is None:
                continue
            img = enhance_image(img)
            img = cv2.resize(img, (size, size))
            out_path = os.path.join(dst, os.path.basename(p))
            cv2.imwrite(out_path, img)
            count += 1

    return count


# ---------------- CLI MODE ----------------
def cli_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Raw images folder path')
    ap.add_argument('--dst', required=True, help='Output folder path')
    ap.add_argument('--size', type=int, default=640, help='Resize dimension (square)')
    args = ap.parse_args()

    n = process_folder(args.src, args.dst, size=args.size)
    print(f"✅ Done! Processed {n} images from '{args.src}' → '{args.dst}'.")


# ---------------- STREAMLIT GUI ----------------
def gui_main():
    import streamlit as st
    st.set_page_config(page_title="🖼️ Image Preprocessor", layout="centered")

    st.title("🖼️ Corn Image Enhancement & Resizer Tool")
    st.caption("Prepare your corn images for YOLO training or model input.")

    src = st.text_input("📂 Source folder (raw images)")
    dst = st.text_input("💾 Destination folder (output)")
    size = st.number_input("📏 Image size (px)", 128, 2048, 640, 32)

    if st.button("Start Processing"):
        if not src or not dst:
            st.warning("Please enter both source and destination folders.")
        elif not os.path.exists(src):
            st.error(f"Source folder '{src}' not found.")
        else:
            with st.spinner("Processing images..."):
                n = process_folder(src, dst, size)
            st.success(f"✅ Done! Processed {n} images from '{src}' → '{dst}'.")

    st.markdown("---")
    st.markdown("💡 *Tip: This tool enhances image brightness and contrast (CLAHE), "
                "then resizes all images to a uniform square size for training.*")


# ---------------- MODE DETECTOR ----------------
def is_running_with_streamlit():
    """Detect Streamlit runtime to show GUI properly."""
    try:
        import streamlit.runtime.scriptrunner as srs
        return srs.get_script_run_ctx() is not None
    except Exception:
        return False


# ---------------- ENTRY POINT ----------------
if __name__ == '__main__':
    # Case A: Started via 'streamlit run data_prep.py'
    if is_running_with_streamlit():
        gui_main()

    # Case B: Started with CLI args
    elif len(sys.argv) > 1:
        cli_main()

    # Case C: Started with plain 'python data_prep.py'
    else:
        try:
            import streamlit.web.cli as stcli
            sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
            sys.exit(stcli.main())
        except Exception as e:
            print("⚠️ Streamlit failed to launch GUI:", e)
            print("Fallback to CLI example:")
            print("   python data_prep.py --src raw --dst processed --size 640")
