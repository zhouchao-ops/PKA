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

The base model is not redistributed. These adapters are distributed under the
FLUX.1 [dev] Non-Commercial License because they are fine-tuned derivatives of
FLUX.1 [dev]. See `LICENSE-FLUX1-DEV.md` and `NOTICE`. Users must obtain the
base model separately and comply with the same license, including its
non-commercial and non-production restrictions.
