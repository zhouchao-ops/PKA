# Subject + Depth LoRA

- Base model: `black-forest-labs/FLUX.1-dev`
- Training recipe: `../../configs/subject_depth.yaml`
- Training: 20,000 optimizer steps, batch size 1, gradient accumulation 4
- Optimizer: Prodigy
- LoRA: rank 4, alpha 4
- Conditions: 256 x 256 subject reference and depth map
- Output: 512 x 512
- Layout: compact PAA `aligned1`
