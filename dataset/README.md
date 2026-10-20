---
pretty_name: PKA-SubjectSpatial-10K
license: apache-2.0
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: train/*.parquet
      - split: test
        path: test/*.parquet
---

# PKA-SubjectSpatial-10K

Dataset for [Rethinking Multi-Condition DiTs: Eliminating Redundant Attention via Position-Alignment and Keyword-Scoping](https://arxiv.org/abs/2602.06850), submitted to IEEE TVCG.

PKA-SubjectSpatial-10K is the curated subject-and-spatial-control subset used
to train the three released PKA Compact = 2 LoRAs. It contains target images,
descriptions, subject references, and multiple spatial conditions for
multi-condition image generation.

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

## Training setup

The released FLUX.1-dev LoRAs are trained for 20,000 optimizer steps with
LoRA rank/alpha 4, batch size 1, gradient accumulation 4, and the Prodigy
optimizer. Conditions are encoded at 256 x 256 for 512 x 512 generation.

## Notes

- The release uses Apache-2.0.
- 75 rows have `description_valid=false`; the flag is retained so downstream
  users can filter them explicitly.
- All current rows use the collection label `collection_1`.
- The dataset repository is
  <https://huggingface.co/datasets/zc1696190340/PKA-SubjectSpatial-10K>.

## Citation

```bibtex
@article{zhou2026rethinking,
  title={Rethinking Multi-Condition DiTs: Eliminating Redundant Attention via Position-Alignment and Keyword-Scoping},
  author={Zhou, Chao and Wei, Tianyi and Chen, Yiling and Zhou, Wenbo and Zhang, Weiming and Yu, Nenghai},
  journal={arXiv preprint arXiv:2602.06850},
  year={2026}
}
```
