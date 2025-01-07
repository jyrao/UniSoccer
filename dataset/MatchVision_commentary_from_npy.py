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

class MatchVisionCommentary_from_npy_Dataset(Dataset):
    def __init__(self, json_file, video_base_dir, npy_dir,
                 tokenizer_name = 'Meta-Llama-3-8B-Instruct', max_token_length =128
                 ):
        self.video_base_dir = video_base_dir
        self.npy_dir = npy_dir

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token_id = 128001
        self.tokenizer.add_tokens(["[PLAYER]","[TEAM]","[COACH]","[REFEREE]","([TEAM])"], special_tokens=True)
        self.max_token_length = max_token_length
        self.multiple_json = isinstance(json_file, list)
        # Load data from JSON file
        if not self.multiple_json:
            with open(json_file, 'r') as file:
                self.data = json.load(file)
        else:
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
        num_retries = 20
        for _ in range(num_retries):
            try:
                video_info = self.data[idx]
                if not self.multiple_json:
                    npy_path = os.path.join(self.npy_dir, video_info['video'].replace(".mp4", ".npy"))
                else:
                    npy_path = video_info['video'].replace(".mp4", ".npy")

                feature = torch.from_numpy(np.load(npy_path)).to('cpu')

                caption = video_info['comments_text_anonymized']

                caption_tokens = self.tokenizer(
                        caption,
                        return_tensors = "pt",
                        max_length=self.max_token_length,
                        truncation=True
                ).input_ids[0]

                return {
                    "features": feature,
                    "caption_tokens": caption_tokens,
                    "caption_text": caption
                }
            except:
                old_idx = idx
                idx = random.randint(0, len(self) - 1)
                # print(f"changed index from {old_idx} to {idx}.")
                continue
    
    def collater(self, instances):
        input_ids = [
            torch.cat((torch.tensor([self.tokenizer.convert_tokens_to_ids("<|begin_of_text|>")]),
                       instance["caption_tokens"],
                       torch.tensor([self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")]))) for instance in instances] # add end token
        labels = copy.deepcopy(input_ids)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"))
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        batch = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")),
            labels=labels,
        )
        batch["caption_text"] = [instance['caption_text'] for instance in instances]
        if 'features' in instances[0]:
            features = [instance['features'] for instance in instances]
            if all(x is not None and x.shape == features[0].shape for x in features):
                batch['features'] = torch.stack(features)
            else:
                batch['features'] = features
        return batch
