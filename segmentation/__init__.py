"""Foreground segmentation for the white-background query renders.

Public entry point: :func:`segment_image`.

    from segmentation import segment_image, SAMSegmenter, load_rgb, imwrite
"""
from .core import (SAMSegmenter, Metrics, coarse_foreground, cv_refine,
                   fill_small_holes, imread, imwrite, largest_components,
                   load_rgb, quality_metrics, sample_prompts, segment_image,
                   shadow_like)

__all__ = [
    "SAMSegmenter", "Metrics", "coarse_foreground", "cv_refine",
    "fill_small_holes", "imread", "imwrite", "largest_components",
    "load_rgb", "quality_metrics", "sample_prompts", "segment_image",
    "shadow_like",
]
