---
pretty_name: PKA-SubjectSpatial-10K
license: apache-2.0
size_categories:
  - 1K<n<10K
---

# PKA-SubjectSpatial-10K

This directory documents the local dataset used for the three released 20K
LoRAs. The dataset files are intentionally not copied into the source-code
release. Publish them as a separate dataset repository only after completing
the rights and provenance review below.

## Local snapshot

- Public release name: `PKA-SubjectSpatial-10K`
- On-disk size: approximately 13 GB
- Format: Parquet, 50 shards per split
- Train: 8,200 rows
- Test: 1,575 rows
- Total: 9,775 rows
- Collection field: all rows are currently labeled `collection_1`
- Valid descriptions: 8,136/8,200 train and 1,564/1,575 test

The public name is **PKA-SubjectSpatial-10K**, which communicates the
subject/spatial-control scope and approximate size. The snapshot contains
9,775 examples.

## Schema

- `image`: target RGB image
- `collection`: source collection label
- `quality_assessment`: composite structure, image quality, and object
  consistency scores
- `description`: item, two descriptions, category, and validity flag
- Control images: `sketch`, `canny`, `depth`, `normal`, `segmentation`,
  `albedo`, and `irradiance`

The local Parquet schema stores the control images as binary values. The
training recipes released here use subject references together with canny or
depth, or canny together with depth.

## Required review before upload

1. Identify the original source and license of every target image.
2. Document how captions and each control image were created, including model
   names, versions, and licenses.
3. Confirm that redistribution, modification, and model training are allowed.
4. Audit personal data, faces, minors, copyrighted characters, trademarks,
   medical content, and NSFW content.
5. Explain the train/test split and remove duplicates or leakage.
6. Decide whether the 75 rows with invalid descriptions should be removed.
7. Replace the placeholder `collection_1` values with useful provenance, or
   explain why source-level provenance cannot be provided.
8. The release uses Apache-2.0, matching the historical dataset card. The
   publisher remains responsible for confirming that it has the rights needed
   to distribute the underlying images, captions, and generated controls.
