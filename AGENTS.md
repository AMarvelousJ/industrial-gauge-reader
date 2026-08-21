# Multi-agent development rules

## Objective

Build a reproducible visual gauge-style classifier on top of the existing
single-class YOLO meter detector. The immediate acceptance target is at least
80% accuracy on the valid rows from `all_set/仪表盘读数标注.md`, with an honest
description of whether the metric is a closed-set or generalization result.

## Source-of-truth boundaries

- `all_set/` is immutable user data. Never rename, move, edit, or delete files
  under it.
- The existing detector remains a single `meter` class. Do not report detector
  mAP as style-classification accuracy.
- Never use a path, filename, or parent directory as an input feature at
  prediction time. The class label may be derived from `Mxx` only while
  preparing supervised training targets and evaluation ground truth.
- Exclude exact Markdown evaluation images from training. Detect and report
  duplicates and malformed Markdown paths instead of silently ignoring them.

## Ownership

- `style_classifier/`: style-classification implementation and generated
  artifacts.
- `tests/style_classifier/`: automated tests for parsing, splitting, inference,
  and metrics.
- Existing `scripts/`: detector pipeline; modify only when an integration test
  proves a change is necessary.

## Validation contract

Every claimed result must include the command, model path, evaluated sample
count, per-style counts, prediction manifest, and confusion matrix or equivalent
class-by-class summary. A successful run must be reproducible from the project
virtual environment and must not depend on hidden local state.
