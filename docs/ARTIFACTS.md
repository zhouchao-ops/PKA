# Release artifact map

| Artifact | Location in this bundle | Recommended public home |
| --- | --- | --- |
| Source code | repository root | GitHub/GitLab repository |
| Subject + canny LoRA | `checkpoints/subject_canny/` | model repository |
| Subject + depth LoRA | `checkpoints/subject_depth/` | model repository |
| Canny + depth LoRA | `checkpoints/canny_depth/` | model repository |
| Training dataset | not copied; documented in `dataset/` | separate dataset repository |
| Training recipes | `configs/` | source repository and model cards |

The original local training data remains outside this bundle. Machine-specific
paths must not appear in public configs or usage examples.
