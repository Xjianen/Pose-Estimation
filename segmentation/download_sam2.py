"""Download the optional SAM 2.1 ONNX weights (int8, CPU-runnable).

    python segmentation/download_sam2.py                 # B+, ~108 MB
    python segmentation/download_sam2.py --variant large   # ~260 MB
    python segmentation/download_sam2.py --mirror https://your-mirror

Files are written to segmentation/models_sam2/ (gitignored). The ONNX exports
are from onnx-community on Hugging Face; hf-mirror.com is tried first because
the project's servers are usually inside mainland China.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

VARIANTS = {
    "tiny": "onnx-community/sam2.1-hiera-tiny-ONNX",
    "small": "onnx-community/sam2.1-hiera-small-ONNX",
    "base-plus": "onnx-community/sam2.1-hiera-base-plus-ONNX",
    "large": "onnx-community/sam2.1-hiera-large-ONNX",
}
FILES = [
    "vision_encoder_quantized.onnx",
    "vision_encoder_quantized.onnx_data",
    "prompt_encoder_mask_decoder_quantized.onnx",
    "prompt_encoder_mask_decoder_quantized.onnx_data",
]
MIRRORS = ["https://hf-mirror.com", "https://huggingface.co"]


def fetch(url, dest, timeout=600):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r  {got/1e6:7.1f}/{total/1e6:.1f} MB", end="",
                      flush=True)
    print()
    if got < 1024:
        tmp.unlink(missing_ok=True)
        raise IOError(f"download too small ({got} bytes): {url}")
    tmp.replace(dest)
    print(f"  -> {dest}  ({got/1e6:.1f} MB, {time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="base-plus")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(__file__), "models_sam2"))
    ap.add_argument("--mirror", default=None,
                    help="base URL prepended to the mirror list")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    repo = VARIANTS[args.variant]
    mirrors = ([args.mirror.rstrip("/")] + MIRRORS) if args.mirror else MIRRORS
    out = Path(args.out_dir)
    for name in FILES:
        dest = out / name
        if dest.is_file() and dest.stat().st_size > 1024 and not args.force:
            print(f"skip {name} ({dest.stat().st_size/1e6:.1f} MB)")
            continue
        ok = False
        for base in mirrors:
            url = f"{base}/{repo}/resolve/main/onnx/{name}"
            try:
                print(f"GET {url}")
                fetch(url, dest)
                ok = True
                break
            except Exception as exc:
                print(f"  failed: {exc}", file=sys.stderr)
        if not ok:
            print(f"could not download {name} from any mirror", file=sys.stderr)
            return 1
    print("done. Use --mode sam2 in run_segmentation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
