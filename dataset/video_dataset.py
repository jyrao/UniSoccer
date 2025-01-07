import torch
import sys
sys.path.append('PATH_TO_FOLDER_OF_THIS_PROJECT')
import json
import os
import random
from einops import rearrange
from torch.utils.data import Dataset
from dataset.video_utils_siglip import read_frames_decord, set_transform

class VideoCaptionDataset(Dataset):
    def __init__(self, json_file, video_base_dir, num_frames=30, sample='rand', 
                 fix_start=None, max_num_frames=-1, trimmed30=False,
                 keywords = [
                    'corner', 'goal', 'injury', 'own goal', 'penalty', 'penalty missed', 'red card', 'second yellow card', 'substitution', 'start of game(half)', 'end of game(half)', 'yellow card', 'throw in', 'free kick', 'saved by goal-keeper', 'shot off target', 'clearance', "lead to corner", 'off-side', 'var', 'foul with no card', 'statistics and summary', 'ball possession', 'ball out of play'
                ],
                sample_num = None,
                require_text = False,
                text_key = "comments_text_anonymized",
                ):
        self.video_base_dir = video_base_dir
        self.num_frames = num_frames
        self.sample = sample
        self.fix_start = fix_start
        self.max_num_frames = max_num_frames
        self.trimmed30 = trimmed30
        self.keywords = keywords
        self.transform = set_transform()
        self.require_text = require_text
        self.text_key = text_key

        self.data = []
        # Load data from JSON file
        for i in range(len(json_file)):
            with open(json_file[i], 'r') as file:
                current_data = json.load(file)
                for item in current_data:
                    item["video"] = os.path.join(video_base_dir[i], item["video"])
                self.data.extend(current_data)
                print(f"File loaded: {json_file[i]}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        num_retries = 50
        for _ in range(num_retries):
            try:
                video_info = self.data[idx]
                video_path = video_info['video']
                caption = self.caption_to_tensor(video_info['caption'])
                # Extract frames using the pre-defined function
                frames, frame_indices, duration = read_frames_decord(
                    video_path, self.num_frames, self.sample, self.fix_start, 
                    self.max_num_frames, self.trimmed30
                )
                frames = torch.cat([self.transform(images=frame, return_tensors="pt")["pixel_values"] for frame in frames], dim=0)
                frames = rearrange(frames, 't c h w -> c t h w')
                if self.require_text:
                    return frames, caption, video_info['video'], video_info['caption'], video_info[self.text_key]
                return frames, caption
            except:
                old_idx = idx
                idx = random.randint(0, len(self) - 1)
                print(f"changed index from {old_idx} to {idx}.")
                continue
    
    def caption_to_tensor(self, caption):
        """
        Converts a caption string to a tensor based on the keywords list.
        The tensor will contain the index of the keyword found in the caption.
        If the caption does not match any keyword, the tensor will contain -1.
        """
        # Initialize the tensor with a default value of -1 (indicating no match)
        caption_index = -1
        for i, keyword in enumerate(self.keywords):
            if keyword == caption:
                caption_index = i
                break
        
        # Convert the index to a tensor
        caption_tensor = torch.tensor(caption_index, dtype=torch.long)
                
        return caption_tensor

class VideoCaptionDataset_Balanced(Dataset):
    def __init__(self, json_file = ["./train_data/json/SoccerNet-v2/classification_train.json"], 
                video_base_dir=["PATH_TO_FOLDER_OF_VIDEO_CLIPS_OF_SOCCERNET_V2"], 
                 sample='rand', 
                 num_frames=30, fix_start=None, max_num_frames=-1, trimmed30=False,
                 keywords=['Penalty', 'Kick-off', 'Shots off target', 'Shots on target', 'Throw-in', 'Ball out of play', 'Foul', 'Direct free-kick', 'Yellow card', 'Goal', 'Clearance', 'Indirect free-kick', 'Offside', 'Corner', 'Yellow->red card', 'Red card', 'Substitution'],
                 sample_num=[500, 2000, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 250, 250, 2500],
                 require_text = False,
                 text_key = "comments_text"
                 ):
        self.video_base_dir = video_base_dir
        self.num_frames = num_frames
        self.sample = sample
        self.fix_start = fix_start
        self.max_num_frames = max_num_frames
        self.trimmed30 = trimmed30
        self.keywords = keywords
        self.sample_num = sample_num
        self.transform = set_transform()
        self.require_text = require_text
        self.text_key = text_key
        
        self.data = []
        # Load data from JSON file
        for i in range(len(json_file)):
            with open(json_file[i], 'r') as file:
                current_data = json.load(file)
                for item in current_data:
                    item["video"] = os.path.join(video_base_dir[i], item["video"])
                self.data.extend(current_data)
                print(f"File loaded: {json_file[i]}")

        # Pre-process data to form a balanced dataset per epoch
        self.preprocess_data()

    def preprocess_data(self):
        self.keyword_to_indices = {keyword: [] for keyword in self.keywords}
        for i, item in enumerate(self.data):
            caption = item['caption']
            for j, keyword in enumerate(self.keywords):
                if keyword == caption:
                    self.keyword_to_indices[keyword].append(i)
        self.shuffle_indices()

    def shuffle_indices(self):
        self.balanced_indices = []
        for keyword, count in zip(self.keywords, self.sample_num):
            available_indices = self.keyword_to_indices[keyword]
            if len(available_indices) >= count:
                sampled_indices = random.sample(available_indices, count)
            else:
                # print(keyword, len(available_indices), count)
                sampled_indices = random.choices(available_indices, k=count)
            self.balanced_indices.extend(sampled_indices)
        random.shuffle(self.balanced_indices)

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        num_retries = 50
        for _ in range(num_retries):
            try:
                actual_idx = self.balanced_indices[idx]
                video_info = self.data[actual_idx]
                video_path = video_info['video']
                caption = self.caption_to_tensor(video_info['caption'])

                frames, frame_indices, duration = read_frames_decord(
                    video_path, self.num_frames, self.sample, self.fix_start, 
                    self.max_num_frames, self.trimmed30
                )
                frames = torch.cat([self.transform(images=frame, return_tensors="pt")["pixel_values"] for frame in frames], dim=0)
                frames = rearrange(frames, 't c h w -> c t h w')
                if self.require_text:
                    return frames, caption, video_info['video'], video_info['caption'], video_info[self.text_key]
                return frames, caption
            except:
                idx = random.randint(0, len(self) - 1)
                continue

    def caption_to_tensor(self, caption):
        caption_index = -1
        for i, keyword in enumerate(self.keywords):
            if keyword == caption:
                caption_index = i
                break
        caption_tensor = torch.tensor(caption_index, dtype=torch.long)
        return caption_tensor

