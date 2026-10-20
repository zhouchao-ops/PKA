# Canny + Depth LoRA

- Base model: `black-forest-labs/FLUX.1-dev`
- Training recipe: `../../configs/canny_depth.yaml`
- Training: 20,000 optimizer steps, batch size 1, gradient accumulation 4
- Optimizer: Prodigy
- LoRA: rank 4, alpha 4
- Conditions: 256 x 256 canny and depth maps
- Output: 512 x 512
- Layout: corrected compact PAA `aligned1`

Public quantitative results for this checkpoint have not yet been added to
this release. Run and document a matching held-out evaluation before release.
