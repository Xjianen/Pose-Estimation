"""Optional SAM 2.1 ONNX backend (onnx-community ONNX exports, int8 quantised).

This exists to make the "would a bigger / newer SAM help?" experiment
reproducible. It is NOT the recommended default: on this dataset SAM2.1-B+
scored *below* the 14 MB SlimSAM and far below the deterministic refine.py
pipeline, because the dominant error is a 1 px rasterisation mismatch rather
than object-level segmentation quality. See README section 9.

Weights are downloaded by ``download_sam2.py`` into ``models_sam2/``.
Interface mirrors ``core.SAMSegmenter`` (``.segment(rgb, coarse) -> mask``),
so it can be dropped into ``segment_image()`` or ``--mode sam2``.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ENCODER = "vision_encoder_quantized.onnx"
DECODER = "prompt_encoder_mask_decoder_quantized.onnx"


class SAM2Segmenter:
    """SAM 2.1 Hiera-B+ ONNX. ~108 MB of int8 weights, CPU-only friendly."""

    def __init__(self, model_dir=None, threads=4):
        import onnxruntime as ort

        model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), "models_sam2")
        enc, dec = os.path.join(model_dir, ENCODER), os.path.join(model_dir, DECODER)
        missing = [p for p in (enc, dec) if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "SAM2 weights missing: " + ", ".join(missing)
                + "\nRun: python segmentation/download_sam2.py")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(threads, os.cpu_count() or 1))
        self.encoder = ort.InferenceSession(enc, opts,
                                            providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(dec, opts,
                                            providers=["CPUExecutionProvider"])

    def segment(self, rgb, coarse):
        try:
            from core import sample_prompts, largest_components, fill_small_holes, _iou
        except ImportError:
            from .core import sample_prompts, largest_components, fill_small_holes, _iou

        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (1024, 1024),
                       interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = np.ascontiguousarray(((x - MEAN) / STD).transpose(2, 0, 1)[None])
        emb = self.encoder.run(None, {"pixel_values": x})

        pts, labs = sample_prompts(coarse, rgb)
        ys, xs = np.nonzero(coarse)
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
        # This ONNX export expects prompts in 1024-space (verified: unscaled
        # points produce empty masks). A box prompt is required for stable
        # multi-candidate behaviour.
        feed = {"input_points": (pts * 2).astype(np.float32)[None, None],
                "input_labels": labs[None, None],
                "input_boxes": (box * 2)[None, None]}
        for i, e in enumerate(emb):
            feed[f"image_embeddings.{i}"] = e
        scores, masks, _ = self.decoder.run(None, feed)

        area_c = max(coarse.sum(), 1)
        best, best_score = None, -1.0
        for k in range(masks.shape[2]):
            a = cv2.resize(masks[0, 0, k].astype(np.float32), (w, h),
                           interpolation=cv2.INTER_LINEAR)
            m = (a > 0).astype(np.uint8)
            ratio = m.sum() / area_c
            if not (0.55 < ratio < 1.45):
                continue
            s = _iou(m, coarse) + 0.2 * float(scores[0, 0, k])
            if s > best_score:
                best_score, best = s, m
        if best is None:
            return None
        best = fill_small_holes(largest_components(best))
        return cv2.morphologyEx(best, cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
