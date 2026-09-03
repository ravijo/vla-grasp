import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100
DATASET_NAME = "pybullet_bottle"


class BottleDataset(Dataset):
    def __init__(self,
                 demo_dir,
                 action_tokenizer,
                 base_tokenizer,
                 image_transform,
                 prompt_builder_fn,
                 dataset_name=DATASET_NAME,
                 ):
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_name = dataset_name

        files = sorted(glob.glob(os.path.join(demo_dir, "episode_*.npz")))
        assert files, f"No demos found in {demo_dir}"
        images, actions, n_traj = [], [], 0
        self.instruction = "pick up the bottle and place it on the tray"
        for f in files:
            d = np.load(f, allow_pickle=True)
            images.append(d["images"])
            actions.append(d["actions"])
            self.instruction = str(d["instruction"])
            n_traj += 1

        # (N, 224, 224, 3) uint8
        self.images = np.concatenate(images, axis=0)
        self.actions = np.concatenate(
            actions, axis=0).astype(np.float32)  # (N, 7)

        # Per-dim normalization bounds (BOUNDS_Q99).
        self.q01 = np.quantile(self.actions, 0.01, axis=0).astype(np.float32)
        self.q99 = np.quantile(self.actions, 0.99, axis=0).astype(np.float32)
        self.dataset_statistics = {
            dataset_name: {
                "action": {
                    "q01": self.q01,
                    "q99": self.q99,
                    "mask": np.ones(7, dtype=bool),
                    "mean": self.actions.mean(0),
                    "std": self.actions.std(0),
                    "min": self.actions.min(0),
                    "max": self.actions.max(0),
                },
                "num_transitions": int(len(self.actions)),
                "num_trajectories": int(n_traj),
            }
        }

    def _normalize(self, a):
        denom = self.q99 - self.q01
        safe = np.where(denom > 1e-6, denom, 1.0)
        norm = np.where(denom > 1e-6, 2.0 * (a - self.q01) / safe - 1.0, 0.0)
        return np.clip(norm, -1.0, 1.0).astype(np.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        action = self._normalize(self.actions[idx])

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {self.instruction.lower()}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(
            prompt_builder.get_prompt(),
            add_special_tokens=True
        ).input_ids
        labels = list(input_ids)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # Only supervise the action tokens (+ stop token).
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
