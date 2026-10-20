#!/usr/bin/env bash
set -euo pipefail

: "${FLUX_MODEL:?Set FLUX_MODEL to a FLUX model ID or local directory}"
: "${OMINI_LORA:?Set OMINI_LORA to the released LoRA file or directory}"

ominicontrol \
  --model "$FLUX_MODEL" \
  --lora "$OMINI_LORA" \
  --prompt "A close-up photo of this item on a wooden table." \
  --keyword "item" \
  --condition subject=examples/subject.png \
  --condition canny=examples/canny.png \
  --output outputs/subject_canny.png

