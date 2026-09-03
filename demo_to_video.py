"""
Convert a collected demo episode (.npz) into a playable MP4

Usage:
    python demo_to_video.py demos/episode_0000.npz --out ep0.mp4 --fps 20 --scale 3
"""

import argparse
import os

import numpy as np
import imageio.v2 as imageio
from PIL import Image


def convert(npz_path, out_path=None, fps=20, scale=2):
    data = np.load(npz_path, allow_pickle=True)
    images = data["images"]  # (T, 224, 224, 3) uint8
    instruction = str(data["instruction"]) if "instruction" in data else ""
    if out_path is None:
        out_path = os.path.splitext(npz_path)[0] + ".mp4"

    h, w = images.shape[1:3]
    out_size = (w * scale, h * scale)

    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
    for frame in images:
        if scale != 1:
            frame = np.asarray(
                Image.fromarray(frame).resize(out_size, Image.NEAREST),
            )
        writer.append_data(frame)
    writer.close()

    print(f"Instruction : {instruction}")
    print(
        f"Frames      : {len(images)}  ({len(images) / fps:.1f}s at {fps} fps)"
    )
    print(f"Saved video : {out_path}  ({out_size[0]}x{out_size[1]})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "npz", help="path to an episode .npz (e.g. demos/episode_0000.npz)")
    ap.add_argument("--out", default=None,
                    help="output .mp4 path (default: next to input)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--scale", type=int, default=2,
                    help="integer upscale factor for visibility")
    args = ap.parse_args()
    convert(args.npz, args.out, args.fps, args.scale)
