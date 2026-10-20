<div align="center">

# PKA

### Rethinking Multi-Condition DiTs: Eliminating Redundant Attention via Position-Alignment and Keyword-Scoping

Chao Zhou · Tianyi Wei · Yiling Chen · Wenbo Zhou · Weiming Zhang · Nenghai Yu

**Submitted to IEEE Transactions on Visualization and Computer Graphics (TVCG)**

[![Paper](https://img.shields.io/badge/arXiv-2602.06850-b31b1b.svg)](https://arxiv.org/abs/2602.06850)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-PKA--SubjectSpatial--10K-yellow)](https://huggingface.co/datasets/zc1696190340/PKA-SubjectSpatial-10K)
[![License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)

</div>

![PKA qualitative results](assets/teaser.png)

PKA is an efficient multi-condition control framework for Diffusion Transformers. It removes redundant image-to-condition interactions while preserving control quality through **Position-Aligned Attention (PAA)**, **Keyword-Scoped Attention (KSA)**, and **Conditional Sensitivity-Aware Sampling (CSAS)**. In our experiments, PKA delivers up to **10.0× faster attention computation** and **5.1× lower attention-module VRAM usage**.

This repository provides our corrected **Compact = 2** implementation for FLUX.1-dev/OminiControl2, training recipes, and three final 20K-step LoRA adapters. For 512 × 512 generation, spatial conditions are encoded at 256 × 256.

## Highlights

- **PAA:** replaces dense spatial-condition attention with position-aligned interactions.
- **KSA:** restricts subject attention to regions associated with the prompt keyword.
- **CSAS:** emphasizes high-sensitivity timesteps during fine-tuning.
- **Multi-condition support:** subject, canny, depth, sketch, normal, segmentation, albedo, and irradiance.
- **Ready-to-use weights:** subject+canny, subject+depth, and canny+depth LoRAs trained for 20,000 steps.

## Method

![Overview of PKA](assets/method_overview.png)

PKA decomposes attention according to condition type. PAA preserves the spatial correspondence between an image token and its control signal; KSA derives a prompt-keyword relevance mask and reuses it in subsequent denoising steps. Condition keys and values are computed once and cached.

## Released models

| LoRA | Conditions | Training recipe |
| --- | --- | --- |
| [`checkpoints/subject_canny`](checkpoints/subject_canny) | Subject + Canny | [`configs/subject_canny.yaml`](configs/subject_canny.yaml) |
| [`checkpoints/subject_depth`](checkpoints/subject_depth) | Subject + Depth | [`configs/subject_depth.yaml`](configs/subject_depth.yaml) |
| [`checkpoints/canny_depth`](checkpoints/canny_depth) | Canny + Depth | [`configs/canny_depth.yaml`](configs/canny_depth.yaml) |

All released adapters use LoRA rank/alpha 4 and were trained for 20,000 optimizer steps with batch size 1, gradient accumulation 4, and Prodigy. Checkpoint hashes are listed in [`checkpoints/README.md`](checkpoints/README.md).

## Installation

Python 3.10+ and a CUDA-capable environment are recommended.

```bash
git clone https://github.com/zhouchao-ops/PKA.git
cd PKA
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For training and tests:

```bash
pip install -e '.[train,test]'
```

FLUX.1-dev is not bundled. Accept its license and authenticate with Hugging Face, or provide a local model directory.

## Inference

Pass each input as `TYPE=IMAGE`; repeat `--condition` to combine controls. Inputs are resized to 256 × 256 by default in this compact release.

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

Use the LoRA matching the requested condition pair. Add `--ksa-threshold 0.4` to enable keyword-scoped subject routing. The default KSA score mode is `softmax`; `--ksa-score-mode raw_logit` is also available for experiments.

## Training

Download [PKA-SubjectSpatial-10K](https://huggingface.co/datasets/zc1696190340/PKA-SubjectSpatial-10K), set the model, dataset, and output paths in a recipe, then run:

```bash
XFL_CONFIG=configs/subject_canny.yaml python -m ominicontrol.train.train
```

Paths may also be supplied through `OMINI_FLUX_PATH`, `OMINI_DATASET_PATH`, and `OMINI_SAVE_PATH`. The released recipes use Compact = 2, corrected `aligned1` PAA, LoRA rank/alpha 4, batch size 1, gradient accumulation 4, Prodigy, and 20,000 optimizer steps.

## Dataset

[PKA-SubjectSpatial-10K](https://huggingface.co/datasets/zc1696190340/PKA-SubjectSpatial-10K) contains 9,775 subject-and-spatial-control examples: 8,200 training samples and 1,575 test samples, distributed as 100 Parquet shards. See [`dataset/README.md`](dataset/README.md) for the schema and usage notes.

## Validation

The package can be checked without downloading the base model:

```bash
python -m compileall -q ominicontrol
pytest -q
python -m build
```

## License

The source code is released under [Apache-2.0](LICENSE). The LoRA weights are FLUX.1-dev derivatives and are distributed under the FLUX.1-dev Non-Commercial License included in each checkpoint directory. The dataset is distributed separately under Apache-2.0. Users must also comply with the license of the FLUX.1-dev base model.

## Citation

If this work is useful in your research, please cite:

```bibtex
@article{zhou2026rethinking,
  title   = {Rethinking Multi-Condition DiTs: Eliminating Redundant Attention via Position-Alignment and Keyword-Scoping},
  author  = {Zhou, Chao and Wei, Tianyi and Chen, Yiling and Zhou, Wenbo and Zhang, Weiming and Yu, Nenghai},
  journal = {arXiv preprint arXiv:2602.06850},
  year    = {2026}
}
```

## Acknowledgements

This implementation builds upon [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), Diffusers, and OminiControl2. We thank their authors and maintainers for releasing their work.
