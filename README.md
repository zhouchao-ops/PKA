# PKA

Official implementation of **Rethinking Multi-Condition DiTs: Eliminating
Redundant Attention via Position-Alignment and Keyword-Scoping**.

PKA reduces redundant image-to-condition attention in multi-condition
Diffusion Transformers. Position-Aligned Attention (PAA) keeps aligned spatial
interactions, Keyword-Scoped Attention (KSA) restricts subject interactions to
keyword-relevant regions, and Conditional Sensitivity-Aware Sampling (CSAS)
prioritizes high-sensitivity training timesteps.

This repository is the **Compact = 2** release built on FLUX.1-dev and
OminiControl2: 512 x 512 generation uses 256 x 256 condition images. It
contains corrected compact PAA `aligned1`, optional KSA, training code, and
three final 20K LoRA adapters.

Authors: Chao Zhou, Tianyi Wei, Yiling Chen, Wenbo Zhou, Weiming Zhang, and
Nenghai Yu. Public repository: <https://github.com/zhouchao-ops/PKA>.

This directory is a release candidate for our composable FLUX control
implementation. It keeps the smallest useful surface: multi-image inference,
corrected compact position-aligned attention, LoRA training, and three
representative two-condition recipes. It also contains the three final 20K
LoRA checkpoints and draft release documentation.

The release does **not** include base-model files, training-data Parquet files,
generated images, notebooks, paper figures, or internal benchmark scripts.

## Supported MVP

- Base model: FLUX (`diffusers==0.31.0`)
- Conditions: subject, canny, depth, sketch, normal, segmentation, albedo,
  and irradiance
- Composition: one or more conditions in a single generation
- Compact conditions: 256 px condition images controlling 512 px outputs
- Layout: corrected `aligned1` only; incompatible experimental layouts are not
  included
- Training recipes: subject+canny, subject+depth, and canny+depth
- Optional KSA subject routing with attention reuse, cached mask indices, and
  batched attention-map reduction

## Included 20K checkpoints

| Checkpoint | Conditions | Recipe |
| --- | --- | --- |
| `checkpoints/subject_canny/` | subject + canny | `configs/subject_canny.yaml` |
| `checkpoints/subject_depth/` | subject + depth | `configs/subject_depth.yaml` |
| `checkpoints/canny_depth/` | canny + depth | `configs/canny_depth.yaml` |

Each adapter is approximately 29 MB, uses LoRA rank/alpha 4, and was trained
for 20,000 optimizer steps with batch size 1 and gradient accumulation 4. See
`checkpoints/README.md` for checksums and details.

## Install

Python 3.10+ and a CUDA-capable environment are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For training and tests:

```bash
pip install -e '.[train,test]'
```

Access to the chosen FLUX base model is handled separately. Accept its license
and authenticate with Hugging Face, or pass a local model directory.

## Inference

Each condition is written as `TYPE=IMAGE`; repeat `--condition` to compose
multiple controls. Condition images are resized to 256×256 by default.

```bash
ominicontrol \
  --model black-forest-labs/FLUX.1-dev \
  --lora checkpoints/subject_canny/pytorch_lora_weights.safetensors \
  --prompt "A close-up photo of this item on a wooden table." \
  --keyword item \
  --condition subject=examples/subject.png \
  --condition canny=examples/canny.png \
  --output outputs/result.png
```

Use a LoRA checkpoint trained for the same condition combination. `--keyword`
is only needed for subject-aware prompting. Run `ominicontrol --help` for all
options.

To enable the optimized keyword-selected attention path, add for example:

```bash
--keyword item --ksa-threshold 0.4
```

The default KSA score mode is `softmax`, with exact CPU reduction. The optional
`--ksa-score-mode raw_logit` and `--ksa-reduction-device cuda` paths are faster
alternatives and should be validated against the checkpoint used for release.

## Changes in 0.2.0

- Reuses attention probabilities instead of repeating the full base softmax.
- Keeps per-head keyword maps on the GPU and transfers them in one batch.
- Caches subject and non-subject indices for subsequent denoising steps.
- Avoids zero-initializing an output that is fully overwritten.
- Keeps corrected `aligned1` as the only packaged condition layout.
- Excludes routing perturbations and diagonal-mass diagnostics because they are
  ablation tooling, not MVP runtime behavior.

## Training

1. Copy the closest recipe from `configs/`.
2. Set `flux_path`, `train.dataset.path`, and `train.save_path`.
3. Ensure the dataset exposes `image`, `description`, `quality_assessment`, and
   the configured spatial-condition columns. Paired images use the original
   OminiControl side-by-side layout.
4. Start training:

```bash
XFL_CONFIG=configs/subject_canny.yaml \
  python -m ominicontrol.train.train
```

The three included recipes use LoRA rank/alpha 4, batch size 1, gradient
accumulation 4, the Prodigy optimizer, and 20,000 optimizer steps. They save
every 2,000 steps. Local paths can be
provided without editing the release configs via `OMINI_FLUX_PATH`,
`OMINI_DATASET_PATH`, and `OMINI_SAVE_PATH`.

Weights are written below `<save_path>/<timestamp>/ckpt/`. Weights and dataset
licenses must be reviewed independently before publishing them.

## Training data

The local training snapshot contains 8,200 train and 1,575 test examples in
100 Parquet shards (approximately 13 GB). It is not embedded in this code
bundle. Read `dataset/README.md` before attempting to publish it: its image
provenance and redistribution rights still require confirmation.

## Validate the package

These checks do not download a base model:

```bash
python -m compileall -q ominicontrol
pytest -q
python -m build
```

GPU inference remains the final release gate and requires a compatible FLUX
model plus the intended LoRA checkpoint.

## Release checklist

- Replace this project name and metadata if the public release uses another
  name.
- Add authors, repository URL, citation, and model-card links.
- Publish model cards describing the training data and limitations.
- Verify every checkpoint against its matching condition recipe.
- Run inference from a fresh environment and retain a known-good sample.
- Tag the commit and build wheel/source archives from this directory only.

The detailed owner checklist is in `docs/RELEASE_CHECKLIST.md`; the artifact
layout is documented in `docs/ARTIFACTS.md`.

## License

Source code is Apache-2.0. The PKA LoRA weights are FLUX.1 [dev] derivatives
and use the FLUX.1 [dev] Non-Commercial License included under `checkpoints/`.
The PKA-SubjectSpatial-10K dataset card declares Apache-2.0. These licenses
apply to separate artifacts and must not be conflated.

## Citation

```bibtex
@article{zhou2026pka,
  title={Rethinking Multi-Condition DiTs: Eliminating Redundant Attention via Position-Alignment and Keyword-Scoping},
  author={Zhou, Chao and Wei, Tianyi and Chen, Yiling and Zhou, Wenbo and Zhang, Weiming and Yu, Nenghai},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2026}
}
```
