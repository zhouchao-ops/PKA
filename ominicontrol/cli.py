"""Command-line inference for OminiControl MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image

from .flux import Condition, generate, seed_everything


SUPPORTED_CONDITIONS = (
    "subject",
    "canny",
    "depth",
    "sketch",
    "normal",
    "segmentation",
    "albedo",
    "irradiance",
)


def condition_spec(value: str) -> tuple[str, Path]:
    """Parse a TYPE=IMAGE condition passed on the command line."""
    condition_type, separator, image_path = value.partition("=")
    if not separator or condition_type not in SUPPORTED_CONDITIONS:
        choices = ", ".join(SUPPORTED_CONDITIONS)
        raise argparse.ArgumentTypeError(
            f"expected TYPE=IMAGE where TYPE is one of: {choices}"
        )
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"condition image does not exist: {path}")
    return condition_type, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an image with one or more OminiControl conditions."
    )
    parser.add_argument("--model", required=True, help="FLUX model ID or local path")
    parser.add_argument("--lora", required=True, help="OminiControl LoRA path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--condition",
        required=True,
        action="append",
        type=condition_spec,
        metavar="TYPE=IMAGE",
        help="repeat for multiple conditions",
    )
    parser.add_argument("--output", type=Path, default=Path("output.png"))
    parser.add_argument("--keyword", help="subject word or phrase in the prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--condition-size", type=int, default=256)
    parser.add_argument("--condition-scale", type=float, default=1.0)
    parser.add_argument(
        "--ksa-threshold",
        type=float,
        help="enable keyword-selected attention with a threshold in [0, 1]",
    )
    parser.add_argument(
        "--ksa-score-mode",
        choices=("softmax", "raw_logit"),
        default="softmax",
    )
    parser.add_argument(
        "--ksa-reduction-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="cpu preserves the reference reduction; cuda is faster",
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.width % 16 or args.height % 16:
        raise SystemExit("--width and --height must be divisible by 16")
    if args.condition_size <= 0 or args.condition_size % 16:
        raise SystemExit("--condition-size must be a positive multiple of 16")
    if args.ksa_threshold is not None and not 0 <= args.ksa_threshold <= 1:
        raise SystemExit("--ksa-threshold must be in [0, 1]")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    pipe.load_lora_weights(args.lora, adapter_name="default")

    conditions = []
    for condition_type, image_path in args.condition:
        image = Image.open(image_path).convert("RGB").resize(
            (args.condition_size, args.condition_size)
        )
        position_delta = (
            (0, -args.condition_size // 16)
            if condition_type == "subject"
            else (0, 0)
        )
        conditions.append(
            Condition(condition_type, image, position_delta=position_delta)
        )

    has_subject = any(item.condition_type == "subject" for item in conditions)
    if args.keyword and not has_subject:
        raise SystemExit("--keyword requires a subject condition")
    if args.ksa_threshold is not None and (not has_subject or not args.keyword):
        raise SystemExit("--ksa-threshold requires a subject condition and --keyword")

    model_config = {
        "union_cond_attn": True,
        "add_cond_attn": False,
        "latent_lora": False,
        "spatial_condition": False,
        "spatial_condition_independent": True,
        "subject_region": has_subject,
        "compact_paa": args.condition_size != args.width,
        "compact_paa_mode": "aligned1",
        "condition_layout": "corrected",
        "keyword": args.keyword,
        "threshold": args.ksa_threshold or 0.0,
        "ksa_enabled": args.ksa_threshold is not None,
        "ksa_score_mode": args.ksa_score_mode,
        "ksa_reuse_attention": True,
        "ksa_reduction_device": args.ksa_reduction_device,
    }
    seed_everything(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    result = generate(
        pipe,
        prompt=args.prompt,
        conditions=conditions,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        model_config=model_config,
        condition_scale=args.condition_scale,
        generator=generator,
    ).images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
