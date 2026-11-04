"""Download the public SlimSAM ONNX weights used by this module.

The whole pair is ~14 MB and no training / account / token is required.
Sources (tried in order):

    https://hf-mirror.com/Xenova/slimsam-77-uniform/resolve/main/onnx/<file>
    https://huggingface.co/Xenova/slimsam-77-uniform/resolve/main/onnx/<file>

Usage:
    python segmentation/download_models.py
    python segmentation/download_models.py --model-dir /data/xje/seg_models
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

try:                                  # python segmentation/download_models.py
    from core import SAM_DECODER, SAM_ENCODER
except ImportError:                   # python -m segmentation.download_models
    from .core import SAM_DECODER, SAM_ENCODER

REPO = "Xenova/slimsam-77-uniform"
FILES = {
    SAM_ENCODER: "onnx/vision_encoder_quantized.onnx",
    SAM_DECODER: "onnx/prompt_encoder_mask_decoder_quantized.onnx",
}
MIRRORS = ["https://hf-mirror.com", "https://huggingface.co"]
MIN_BYTES = 1_000_000          # an LFS pointer / error page is only a few bytes


def _fetch(url, dest, timeout=300):
    with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, t0 = 0, time.time()
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(
                    f"\r  {done / 1e6:6.1f} / {total / 1e6:6.1f} MB "
                    f"({100 * done / total:5.1f}%)")
                sys.stdout.flush()
        sys.stdout.write("\n")
    return done, time.time() - t0


def download(model_dir, mirror_order=None):
    os.makedirs(model_dir, exist_ok=True)
    mirrors = mirror_order or MIRRORS
    for fname, remote in FILES.items():
        dest = os.path.join(model_dir, fname)
        if os.path.isfile(dest) and os.path.getsize(dest) >= MIN_BYTES:
            print(f"[skip] {fname} ({os.path.getsize(dest) / 1e6:.1f} MB)")
            continue
        ok = False
        for base in mirrors:
            url = f"{base}/{REPO}/resolve/main/{remote}"
            print(f"[get ] {url}")
            try:
                n, dt = _fetch(url, dest)
                if n < MIN_BYTES:
                    raise RuntimeError(
                        f"only {n} bytes -- likely an LFS pointer or rate-limit page")
                print(f"       saved {n / 1e6:.1f} MB in {dt:.0f}s")
                ok = True
                break
            except Exception as exc:
                print(f"       failed: {type(exc).__name__}: {str(exc)[:120]}")
                if os.path.isfile(dest):
                    os.remove(dest)
        if not ok:
            raise RuntimeError(f"could not download {fname} from any mirror")
    print(f"\nDone. Weights are in {os.path.abspath(model_dir)}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.path.join(here, "models"),
                    help="where to store the ONNX files")
    ap.add_argument("--mirror", default=None,
                    help="custom base URL, e.g. https://hf-mirror.com")
    args = ap.parse_args()
    order = ([args.mirror.rstrip("/")] + MIRRORS) if args.mirror else None
    download(args.model_dir, order)


if __name__ == "__main__":
    main()
