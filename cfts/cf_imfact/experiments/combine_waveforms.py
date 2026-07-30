"""Combine the FaultDetectionA and FruitFlies compare-waveform plots into one
side-by-side image.

Reads the two `waveforms.png` figures already produced by
`faultdetectiona/compare_faultdetectiona.py` and `fruitflies/compare_fruitflies.py`
and stitches them horizontally — no re-computation of counterfactuals needed.

Usage:
    python combine_waveforms.py
    python combine_waveforms.py --out-dir ./results
    python combine_waveforms.py \
        --faultdetectiona-image ./results/faultdetectiona_compare/waveforms.png \
        --fruitflies-image ./results/fruitflies_compare/waveforms.png \
        --out ./results/combined_compare/waveforms_side_by_side.png
"""

from __future__ import annotations

import argparse
import os

from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def combine_side_by_side(left_path: str, right_path: str, out_path: str, gap: int = 24) -> None:
    if not os.path.exists(left_path):
        raise FileNotFoundError(f"FaultDetectionA waveform image not found: {left_path}")
    if not os.path.exists(right_path):
        raise FileNotFoundError(f"FruitFlies waveform image not found: {right_path}")

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    height = max(left.height, right.height)

    def _pad_to_height(img: Image.Image, target_height: int) -> Image.Image:
        if img.height == target_height:
            return img
        canvas = Image.new("RGB", (img.width, target_height), (255, 255, 255))
        canvas.paste(img, (0, 0))
        return canvas

    left = _pad_to_height(left, height)
    right = _pad_to_height(right, height)

    combined = Image.new("RGB", (left.width + gap + right.width, height), (255, 255, 255))
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + gap, 0))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    combined.save(out_path)
    print(f"Saved combined waveform plot: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine FaultDetectionA and FruitFlies compare-waveform plots side by side."
    )
    parser.add_argument("--out-dir", type=str, default=os.path.join(SCRIPT_DIR, "results"),
                        help="Root results directory (default: ./results). Used to derive default "
                             "input/output paths unless overridden.")
    parser.add_argument("--faultdetectiona-image", type=str, default=None,
                        help="Path to the FaultDetectionA waveforms.png "
                             "(default: <out-dir>/faultdetectiona_compare/waveforms.png)")
    parser.add_argument("--fruitflies-image", type=str, default=None,
                        help="Path to the FruitFlies waveforms.png "
                             "(default: <out-dir>/fruitflies_compare/waveforms.png)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for the combined image "
                             "(default: <out-dir>/combined_compare/waveforms_side_by_side.png)")
    return parser.parse_args()


def main():
    args = parse_args()
    left_path = args.faultdetectiona_image or os.path.join(args.out_dir, "faultdetectiona_compare", "waveforms.png")
    right_path = args.fruitflies_image or os.path.join(args.out_dir, "fruitflies_compare", "waveforms.png")
    out_path = args.out or os.path.join(args.out_dir, "combined_compare", "waveforms_side_by_side.png")
    combine_side_by_side(left_path, right_path, out_path)


if __name__ == "__main__":
    main()
