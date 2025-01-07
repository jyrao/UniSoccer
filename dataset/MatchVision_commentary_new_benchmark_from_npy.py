import sys, os
sys.path.append('PATH_TO_FOLDER_OF_THIS_PROJECT')
import torch
import json
from einops import rearrange
from torch.utils.data import Dataset
from dataset.video_utils_siglip import read_frames_decord, set_transform
from transformers import AutoTokenizer
import copy
from torch.utils.data import DataLoader, random_split
import numpy as np
import random

IGNORE_INDEX = -100

class MatchVisionCommentary_new_benchmark_from_npy_Dataset(Dataset):
    def __init__(self, json_file, video_base_dir, 
                 num_frames=30, sample='middle', fix_start=None, max_num_frames=-1, trimmed30=False,
                 tokenizer_name = 'Meta-Llama-3-8B-Instruct', max_token_length =128
                 ):
        self.video_base_dir = video_base_dir
        # self.npy_dir = npy_dir

        self.num_frames = num_frames
        self.sample = sample
        self.fix_start = fix_start
        self.max_num_frames = max_num_frames
        self.trimmed30 = trimmed30
        self.transform = set_transform()

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
                        item["video"] = os.path.join(video_base_dir[i], item["video"])
                        # print(item["video"])
                    self.data.extend(current_data)
                    print(f"File loaded: {json_file[i]}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_info = self.data[idx]
        if not self.multiple_json:
            video_path = os.path.join(self.video_base_dir, video_info['video'])
        else:
            video_path = video_info['video']

        frames, frame_indices, duration = read_frames_decord(
            video_path, self.num_frames, self.sample, self.fix_start, 
            self.max_num_frames, self.trimmed30
        )
        frames = torch.cat([self.transform(images=frame, return_tensors="pt")["pixel_values"] for frame in frames], dim=0)
        frames = rearrange(frames, 't c h w -> c t h w')

        caption = video_info['comments_text_anonymized']

        caption_tokens = self.tokenizer(
                caption,
                return_tensors = "pt",
                max_length=self.max_token_length,
                truncation=True
        ).input_ids[0]

        return {
            "frames": frames,
            "caption_tokens": caption_tokens,
            "caption_text": caption,
            "video_path": video_path
        }
    
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
        batch["video_path"] = [instance['video_path'] for instance in instances]
        if 'frames' in instances[0]:
            features = [instance['frames'] for instance in instances]
            if all(x is not None and x.shape == features[0].shape for x in features):
                batch['frames'] = torch.stack(features)
            else:
                batch['frames'] = features
        return batch
