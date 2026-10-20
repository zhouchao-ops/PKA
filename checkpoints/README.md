# OminiControl MVP LoRA checkpoints

These are the final 20,000-optimizer-step LoRA checkpoints trained with the
matching recipes in `../configs/`.

| Directory | Conditions | File size | LoRA rank / alpha |
| --- | --- | ---: | ---: |
| `subject_canny/` | subject + canny | 29,066,872 bytes | 4 / 4 |
| `subject_depth/` | subject + depth | 29,066,872 bytes | 4 / 4 |
| `canny_depth/` | canny + depth | 29,066,872 bytes | 4 / 4 |

All three use FLUX.1-dev, corrected condition layout, compact PAA `aligned1`,
batch size 1, gradient accumulation 4, Prodigy, and 20,000 optimizer steps.
Each file contains 686 tensors in safetensors format. Verify downloads with:

```bash
cd checkpoints
sha256sum -c SHA256SUMS
```

The base model is not redistributed. Users must obtain it separately and
comply with its license. Before publishing these adapters, the project owner
must choose and state a weight license that is compatible with the base model
and training data.
