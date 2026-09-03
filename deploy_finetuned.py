import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio

# Make sure OpenVLA files are accessible
# Set the REPO_ROOT to the OpenVLA repo root directory, and add it to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoModelForVision2Seq, AutoProcessor

from pick_place_env import INSTRUCTION, PickPlaceEnv
from bottle_dataset import DATASET_NAME

PROMPT = f"In: What action should the robot take to {INSTRUCTION.lower()}?\nOut:"


def find_latest_checkpoint():
    runs = sorted(glob.glob(os.path.join(
        os.path.dirname(__file__), "runs", "*")))
    runs = [r for r in runs if os.path.isfile(
        os.path.join(r, "dataset_statistics.json"))]
    assert runs, "No checkpoint with dataset_statistics.json found under ravi/runs/"
    return runs[-1]


def load_model(checkpoint):
    print(f"[deploy] loading fine-tuned checkpoint: {checkpoint}")
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # avoids the eager KV-cache bug
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")
    # Load action un-normalization stats (as experiments/robot/openvla_utils does)
    with open(os.path.join(checkpoint, "dataset_statistics.json")) as f:
        model.norm_stats = json.load(f)
    return model, processor


def run(checkpoint, episodes=20, max_steps=120, seed=0, video_path=None):
    model, processor = load_model(checkpoint)
    env = PickPlaceEnv(gui=False, seed=seed)

    successes = 0
    best_frames = None
    for ep in range(episodes):
        obs = env.reset()
        frames = [obs["image"]]
        success = False
        for t in range(max_steps):
            inputs = processor(
                PROMPT,
                Image.fromarray(obs["image"])
            ).to("cuda", dtype=torch.bfloat16)
            action = model.predict_action(
                **inputs,
                unnorm_key=DATASET_NAME,
                do_sample=False,
            )
            action = np.asarray(action, dtype=float)
            obs, r, done, _ = env.step(action)
            frames.append(obs["image"])
            if done:
                success = True
                break
        successes += int(success)
        print(f"  episode {ep:2d}: success={success} (steps={t + 1})")
        # keep frames from the first success (good for the video)
        if success and best_frames is None:
            best_frames = frames
    env.close()

    if best_frames is None:
        best_frames = frames  # no success; show the last attempt
    if video_path is None:
        video_path = os.path.join(
            os.path.dirname(__file__),
            "videos",
            "finetuned_rollout.mp4",
        )
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = imageio.get_writer(video_path, fps=20, macro_block_size=1)
    for f in best_frames:
        writer.append_data(
            np.asarray(Image.fromarray(f).resize((448, 448), Image.NEAREST)),
        )
    writer.close()

    print("=" * 60)
    print(
        f"Fine-tuned OpenVLA success rate: {successes}/{episodes} = {100 * successes / episodes:.0f}%"
    )
    print(f"Rollout video saved to: {video_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="path to run dir (default: latest under ravi/runs/)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ckpt = args.checkpoint or find_latest_checkpoint()
    run(ckpt, args.episodes, args.max_steps, args.seed)
