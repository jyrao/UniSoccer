import sys, os
import torch
import json
from einops import rearrange
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import copy
from torch.utils.data import DataLoader, random_split
import numpy as np
import random

IGNORE_INDEX = -100

class MatchVisionClassification_from_npy_Dataset(Dataset):
    def __init__(self, json_file, npy_dir,
                 max_token_length =128,
                 keywords = [
                    'corner', 'goal', 'injury', 'own goal', 'penalty', 'penalty missed', 'red card', 'second yellow card', 'substitution', 'start of game(half)', 'end of game(half)', 'yellow card', 'throw in', 'free kick', 'saved by goal-keeper', 'shot off target', 'clearance', "lead to corner", 'off-side', 'var', 'foul with no card', 'statistics and summary', 'ball possession', 'ball out of play'
                 ],
                 ):
        self.npy_dir = npy_dir
        self.keywords = keywords

        self.data = []
        for i in range(len(json_file)):
            # Load data from JSON file
            with open(json_file[i], 'r') as file:
                current_data = json.load(file)
                for item in current_data:
                    item["video"] = os.path.join(npy_dir[i], item["video"])
                self.data.extend(current_data)
                print(f"File loaded: {json_file[i]}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        num_retries = 10
        for _ in range(num_retries):
            try:
                video_info = self.data[idx]
                npy_path = video_info['video'].replace(".mp4", ".npy")
                feature = torch.from_numpy(np.load(npy_path)).to('cpu')

                caption = video_info['caption']
                caption_tensor = self.caption_to_tensor(caption)

                return feature, caption_tensor
            except:
                old_idx = idx
                idx = random.randint(0, len(self) - 1)
                print(f"changed index from {old_idx} to {idx}.")
                continue

    def collater(self, instances):
        
        features = [instance[0] for instance in instances]
        features = torch.stack(features)
        caption_tensor = [instance[1] for instance in instances]
        caption_tensor = torch.stack(caption_tensor)
        return features, caption_tensor

    def caption_to_tensor(self, caption):
        caption_index = -1
        for i, keyword in enumerate(self.keywords):
            if keyword == caption:
                caption_index = i
                break
        caption_tensor = torch.tensor(caption_index, dtype=torch.long)
        return caption_tensor