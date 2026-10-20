# Subject + Canny LoRA

- Base model: `black-forest-labs/FLUX.1-dev`
- Training recipe: `../../configs/subject_canny.yaml`
- Training: 20,000 optimizer steps, batch size 1, gradient accumulation 4
- Optimizer: Prodigy
- LoRA: rank 4, alpha 4
- Conditions: 256 x 256 subject reference and canny map
- Output: 512 x 512
- Layout: corrected compact PAA `aligned1`

## 300-case evaluation

| Metric | Value |
| --- | ---: |
| FID (lower is better) | 55.7867 |
| SSIM | 0.5183 |
| Canny F1 | 0.3128 |
| CLIP-I | 0.9462 |
| DINOv2 | 0.9186 |
| CLIP-T | 0.3476 |

The evaluation summary is included as `eval_300_summary.json`. Document the
evaluation dataset and metric implementations before presenting these values
as a public benchmark.
