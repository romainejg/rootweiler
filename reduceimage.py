#!/usr/bin/env python3
"""
Shrink images to ~200–400 KB each with folder prompts.
- Preserves originals (reads from input folder, writes to output folder).
- Adapts JPEG quality + downscales if needed to hit size.
- Converts non-JPEGs to JPEG (transparent pixels are flattened to white).
- Recurses into subfolders and keeps structure.
"""

import io, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Deps ----
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# ---- UI folder prompts ----
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TK_OK = True
except Exception:
    TK_OK = False

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic", ".heif"}
TARGET_MIN_KB = 200
TARGET_MAX_KB = 400
TARGET_MID_KB = 300  # aim point
START_MAX_EDGE = 2200  # start long edge; will shrink further if needed
MIN_MAX_EDGE = 1200    # won't shrink below this unless absolutely needed
KEEP_EXIF = False      # EXIF adds size; off for tighter targets
WORKERS = max(4, (os.cpu_count() or 4) // 2)

def select_folders():
    if not TK_OK:
        print("tkinter not available; pass folders via CLI instead.")
        return None, None
    root = tk.Tk()
    root.withdraw()
    root.update()
    in_dir = filedialog.askdirectory(title="Select INPUT folder (original images)")
    if not in_dir:
        return None, None
    out_dir = filedialog.askdirectory(title="Select OUTPUT folder (resized copies)")
    root.update()
    root.destroy()
    return in_dir, out_dir

def bytes_for_save(img: Image.Image, quality: int, exif_bytes=None) -> bytes:
    buf = io.BytesIO()
    params = {"format": "JPEG", "quality": quality, "optimize": True, "progressive": True}
    if KEEP_EXIF and exif_bytes:
        params["exif"] = exif_bytes
    img.save(buf, **params)
    return buf.getvalue()

def flatten_to_rgb(img: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    # Remove alpha/transparency for JPEG
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        base = Image.new("RGB", img.size, bg)
        base.paste(img, mask=img.split()[-1] if img.mode != "P" else img)
        return base
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def resize_long_edge(img: Image.Image, max_edge: int) -> Image.Image:
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return img
    scale = max_edge / float(long_edge)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return img.resize(new_size, Image.LANCZOS)

def hit_target_size(img_rgb: Image.Image, exif_bytes, max_edge_start=START_MAX_EDGE):
    """
    Tries to produce JPEG bytes in 200–400 KB by:
      1) Downscaling to max_edge_start (and reducing if necessary)
      2) Binary-searching JPEG quality
    Returns (jpeg_bytes, final_edge, final_quality)
    """
    max_edge = max_edge_start
    attempt = 0

    while True:
        attempt += 1
        candidate = resize_long_edge(img_rgb, max_edge)

        # Binary search quality to hit around TARGET_MID_KB within bounds
        q_lo, q_hi = 35, 92
        best_bytes, best_q, best_delta = None, None, None

        while q_lo <= q_hi:
            q = (q_lo + q_hi) // 2
            data = bytes_for_save(candidate, q, exif_bytes)
            size_kb = len(data) / 1024.0
            delta = abs(size_kb - TARGET_MID_KB)

            if best_delta is None or delta < best_delta:
                best_bytes, best_q, best_delta = data, q, delta

            if size_kb > TARGET_MAX_KB:
                # too big -> lower quality
                q_hi = q - 1
            elif size_kb < TARGET_MIN_KB:
                # too small -> raise quality
                q_lo = q + 1
            else:
                # within band
                return data, max_edge, q

        # If best try still above 400 KB, shrink dimensions and try again
        if (len(best_bytes) / 1024.0) > TARGET_MAX_KB and max_edge > MIN_MAX_EDGE:
            max_edge = max(int(max_edge * 0.9), MIN_MAX_EDGE)  # shrink 10% and retry
            continue

        # If best is < 200 KB, we already used highest reasonable quality; accept best
        return best_bytes, max_edge, best_q

def process_one(path: Path, in_root: Path, out_root: Path):
    rel = path.relative_to(in_root)
    out_path = out_root / rel.with_suffix(".jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            # convert to RGB / flatten alpha
            im = flatten_to_rgb(im)
            exif_bytes = im.info.get("exif") if KEEP_EXIF else None

            jpeg_bytes, used_edge, used_q = hit_target_size(im, exif_bytes, START_MAX_EDGE)
            with open(out_path, "wb") as f:
                f.write(jpeg_bytes)

        return ("ok", str(path), str(out_path), used_edge, used_q, round(len(jpeg_bytes)/1024))
    except Exception as e:
        return ("err", str(path), str(out_path), None, None, str(e))

def collect_files(in_root: Path):
    return [p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]

def main():
    # CLI override: python script.py INPUT OUTPUT
    if len(sys.argv) >= 3:
        in_dir = sys.argv[1]
        out_dir = sys.argv[2]
    else:
        in_dir, out_dir = select_folders()
        if not in_dir or not out_dir:
            print("No folders selected. Exiting.")
            sys.exit(1)

    in_root = Path(in_dir)
    out_root = Path(out_dir)
    if not in_root.exists():
        print("Input folder does not exist.")
        sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)

    files = collect_files(in_root)
    if not files:
        print("No supported images found.")
        sys.exit(1)

    print(f"Found {len(files)} images. Writing JPEG copies to: {out_root}")
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(process_one, p, in_root, out_root) for p in files]
        if HAS_TQDM:
            for fut in tqdm(as_completed(futs), total=len(futs), unit="img"):
                results.append(fut.result())
        else:
            for fut in as_completed(futs):
                results.append(fut.result())

    ok = [r for r in results if r[0] == "ok"]
    err = [r for r in results if r[0] == "err"]

    if ok:
        avg_kb = sum(r[5] for r in ok) / len(ok)
        print(f"\nDone. OK: {len(ok)}, Errors: {len(err)}")
        print(f"Average output size: ~{avg_kb:.0f} KB")
        # Show a few samples
        print("\nSamples:")
        for r in ok[:5]:
            _, src, dst, edge, q, kb = r
            print(f"- {Path(src).name} -> {Path(dst).name} | {kb} KB | edge≤{edge}px | q={q}")
    else:
        print("No files processed successfully.")

    if err:
        print("\nErrors (first 10):")
        for r in err[:10]:
            _, src, dst, _, _, msg = r
            print(f"- {src} -> {dst}: {msg}")

if __name__ == "__main__":
    main()
