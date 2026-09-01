import argparse
import json
import os
import time

import numpy as np

from pick_place_env import INSTRUCTION, ExpertPolicy, PickPlaceEnv

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos")


def state_vec(obs):
    return np.concatenate([obs["ee_pos"], [obs["finger_width"]]]).astype(np.float32)


def collect(n_target=100, max_steps=200, seed=0):
    os.makedirs(DEMO_DIR, exist_ok=True)
    env = PickPlaceEnv(gui=False, seed=seed)
    expert = ExpertPolicy()

    saved, attempts, total_transitions, lengths = 0, 0, 0, []
    t0 = time.time()
    while saved < n_target:
        attempts += 1
        obs = env.reset()
        expert.reset()
        images, actions, states = [], [], []

        for t in range(max_steps):
            a = expert.act(obs)
            images.append(obs["image"].copy())  # image before acting
            actions.append(a.astype(np.float32))
            states.append(state_vec(obs))
            obs, _, _, _ = env.step(a)
            if expert.done:  # run through the full release, then judge success
                break

        if not env._success(obs):
            continue  # drop failed episode

        ep_path = os.path.join(DEMO_DIR, f"episode_{saved:04d}.npz")
        np.savez_compressed(
            ep_path,
            images=np.asarray(images, dtype=np.uint8),  # (T, 224, 224, 3)
            actions=np.asarray(actions, dtype=np.float32),  # (T, 7)
            states=np.asarray(states, dtype=np.float32),  # (T, 4)
            instruction=INSTRUCTION,
        )
        saved += 1
        total_transitions += len(actions)
        lengths.append(len(actions))
        if saved % 10 == 0 or saved == n_target:
            rate = 100.0 * saved / attempts
            print(
                f"  saved {saved:3d}/{n_target}  (attempts={attempts}, "
                f"expert success={rate:.0f}%, {time.time() - t0:.0f}s elapsed)"
            )

    env.close()

    meta = {
        "instruction": INSTRUCTION,
        "num_episodes": saved,
        "num_transitions": total_transitions,
        "avg_episode_len": float(np.mean(lengths)),
        "action_dim": 7,
        "action_layout": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        "image_shape": [224, 224, 3],
        "state_dim": 4,
        "state_layout": ["ee_x", "ee_y", "ee_z", "gripper_width"],
        "expert_success_rate": 100.0 * saved / attempts,
        "collection_seconds": time.time() - t0,
    }
    with open(os.path.join(DEMO_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print(
        f"Done: {saved} episodes, {total_transitions} transitions "
        f"(avg len {meta['avg_episode_len']:.1f}), "
        f"expert success {meta['expert_success_rate']:.0f}%"
    )
    print(f"Saved to {DEMO_DIR}/ (+ metadata.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100,
                    help="number of successful episodes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    collect(n_target=args.n, seed=args.seed)
